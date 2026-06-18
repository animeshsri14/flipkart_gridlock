import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def get_hour_bucket(hour: int) -> str:
    """Bucket hour of day according to DESIGN.md section 4.2."""
    if 6 <= hour < 10 or 17 <= hour < 22:
        return 'peak'
    elif 10 <= hour < 17:
        return 'off-peak'
    else:
        return 'night'

class Forecaster:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initializes the forecaster. df should contain:
        'corridor', 'event_cause', 'priority', 'hour_of_day', 'clearance_minutes'
        """
        self.global_median = 48.4
        self.level1 = {}
        self.level2 = {}
        self.level3 = {}
        self.level4 = {}
        
        if df is not None and not df.empty:
            df = df.copy()
            df['hour_bucket'] = df['hour_of_day'].apply(get_hour_bucket)
            
            # Drop NaN clearance_minutes
            df = df.dropna(subset=['clearance_minutes'])
            
            self.level1 = self._build_lookup(df, ['corridor', 'event_cause', 'priority', 'hour_bucket'])
            self.level2 = self._build_lookup(df, ['corridor', 'event_cause', 'priority'])
            self.level3 = self._build_lookup(df, ['event_cause', 'priority'])
            self.level4 = self._build_lookup(df, ['event_cause'])
            self.global_median = df['clearance_minutes'].median()

    def _build_lookup(self, df: pd.DataFrame, keys: list) -> dict:
        lookup = {}
        grouped = df.groupby(keys)['clearance_minutes']
        for name, group in grouped:
            n = len(group)
            if n >= 5:
                lookup_key = name if isinstance(name, tuple) else (name,)
                lookup[lookup_key] = {
                    'p25': float(group.quantile(0.25)),
                    'p50': float(group.median()),
                    'p75': float(group.quantile(0.75)),
                    'p95': float(group.quantile(0.95)),
                    'n': n
                }
        return lookup

    def get_forecast(self, corridor: str, event_cause: str, priority: str, hour: int, 
                     feedback_mean: Optional[float] = None, feedback_n: int = 0) -> Dict[str, Any]:
        """
        Returns the forecasted clearance time based on hierarchical lookup and statistical shrinkage.
        """
        hour_bucket = get_hour_bucket(hour)
        
        keys1 = (corridor, event_cause, priority, hour_bucket)
        keys2 = (corridor, event_cause, priority)
        keys3 = (event_cause, priority)
        keys4 = (event_cause,)
        
        if keys1 in self.level1:
            result = self.level1[keys1]
            level = 1
        elif keys2 in self.level2:
            result = self.level2[keys2]
            level = 2
        elif keys3 in self.level3:
            result = self.level3[keys3]
            level = 3
        elif keys4 in self.level4:
            result = self.level4[keys4]
            level = 4
        else:
            result = {
                'p25': self.global_median,
                'p50': self.global_median,
                'p75': self.global_median,
                'p95': self.global_median,
                'n': 0
            }
            level = 5
            
        p50 = result['p50']
        n_lookup = result['n']
        
        # Shrinkage towards feedback if available
        if feedback_mean is not None and feedback_n >= 3:
            alpha = n_lookup / (n_lookup + feedback_n) if (n_lookup + feedback_n) > 0 else 0
            adjusted_median = (alpha * p50) + ((1 - alpha) * feedback_mean)
        else:
            adjusted_median = p50
            
        return {
            'p25': result['p25'],
            'p50': adjusted_median,
            'p75': result['p75'],
            'p95': result['p95'],
            'n_lookup': n_lookup,
            'lookup_level': level,
            'adjusted_by_feedback': feedback_n >= 3
        }

def evaluate_forecaster(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluates the forecaster using a time-based 80/20 split.
    Requires 'start_datetime' in the DataFrame.
    """
    df = df.copy()
    df = df.dropna(subset=['clearance_minutes', 'start_datetime'])
    df = df.sort_values('start_datetime')
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    forecaster = Forecaster(train_df)
    
    y_true = test_df['clearance_minutes'].values
    y_pred = []
    
    for _, row in test_df.iterrows():
        pred = forecaster.get_forecast(
            corridor=row.get('corridor', 'Non-corridor'),
            event_cause=row.get('event_cause', 'others'),
            priority=row.get('priority', 'Low'),
            hour=row.get('hour_of_day', 12)
        )
        y_pred.append(pred['p50'])
        
    y_pred = np.array(y_pred)
    
    mae_model = np.median(np.abs(y_true - y_pred))
    mae_baseline = np.median(np.abs(y_true - forecaster.global_median))
    
    return {
        'mae_model': float(mae_model),
        'mae_baseline': float(mae_baseline),
        'test_n': len(test_df)
    }
