import pandas as pd
import numpy as np
import json
import streamlit as st
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime
import os

DATA_FILE = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

@dataclass(frozen=True)
class TrafficEvent:
    id: str
    event_type: str           # 'planned' | 'unplanned'
    event_cause: str
    priority: str             # 'High' | 'Low'
    requires_road_closure: bool
    start_dt: datetime
    hour_of_day: int          # 0–23, extracted
    day_of_week: int          # 0=Mon, 6=Sun
    latitude: float
    longitude: float
    end_latitude: Optional[float]   # None if sentinel
    end_longitude: Optional[float]
    corridor: str
    zone: Optional[str]
    junction: Optional[str]
    veh_type: Optional[str]
    route_path: Optional[List[Tuple[float, float]]]
    clearance_minutes: Optional[float]  # None if not computable
    status: str

@st.cache_data
def load_raw_data() -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
    if not os.path.exists(file_path):
        # Return empty dataframe if file not present, helps with initial loads
        return pd.DataFrame()
        
    df = pd.read_csv(file_path, na_values=["NULL", "NaN", ""])
    
    # Coercions
    if 'start_datetime' in df.columns:
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    if 'end_datetime' in df.columns:
        df['end_datetime'] = pd.to_datetime(df['end_datetime'], errors='coerce')
    if 'closed_datetime' in df.columns:
        df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
    if 'resolved_datetime' in df.columns:
        df['resolved_datetime'] = pd.to_datetime(df['resolved_datetime'], errors='coerce')
    
    # Lat/Lon coercion and bbox validation
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        bbox_mask = (df['latitude'].between(12.7, 13.3)) & (df['longitude'].between(77.3, 77.9))
        df.loc[~bbox_mask, ['latitude', 'longitude']] = np.nan
    
    # End Lat/Lon coercion
    if 'endlatitude' in df.columns and 'endlongitude' in df.columns:
        df['endlatitude'] = pd.to_numeric(df['endlatitude'], errors='coerce')
        df['endlongitude'] = pd.to_numeric(df['endlongitude'], errors='coerce')
        df.loc[df['endlatitude'] == 0, 'endlatitude'] = np.nan
        df.loc[df['endlongitude'] == 0, 'endlongitude'] = np.nan
    
    # Priority, Road closure
    if 'priority' in df.columns:
        df['priority'] = df['priority'].astype('category')
    if 'requires_road_closure' in df.columns:
        df['requires_road_closure'] = df['requires_road_closure'].astype(str).str.lower() == 'true'
    
    # Veh type, Corridor
    if 'veh_type' in df.columns:
        df['veh_type'] = df['veh_type'].astype('category')
    if 'corridor' in df.columns:
        df['corridor'] = df['corridor'].fillna('Non-corridor').astype('category')
    
    # Clearance time logic
    if 'resolved_datetime' in df.columns and 'closed_datetime' in df.columns and 'start_datetime' in df.columns:
        df['clearance_end'] = df['resolved_datetime'].combine_first(df['closed_datetime'])
        df['clearance_minutes'] = (df['clearance_end'] - df['start_datetime']).dt.total_seconds() / 60.0
    
    # Route path parsing
    def parse_route(path_str):
        if pd.isna(path_str) or not isinstance(path_str, str):
            return None
        try:
            return json.loads(path_str)
        except json.JSONDecodeError:
            return None
            
    if 'route_path' in df.columns:
        df['parsed_route_path'] = df['route_path'].apply(parse_route)
    else:
        df['parsed_route_path'] = None
        
    return df

@st.cache_data
def load_events() -> List[TrafficEvent]:
    df = load_raw_data()
    if df.empty:
        return []
        
    events = []
    for _, row in df.iterrows():
        start_dt = row.get('start_datetime')
        if pd.isna(start_dt):
            continue
            
        hour_of_day = start_dt.hour
        day_of_week = start_dt.dayofweek
        
        cm = row.get('clearance_minutes')
        clearance_minutes = float(cm) if not pd.isna(cm) else None
        
        event = TrafficEvent(
            id=str(row.get('_id', row.name)),
            event_type=str(row.get('event_type', 'unplanned')),
            event_cause=str(row.get('event_cause', 'unknown')),
            priority=str(row.get('priority', 'Low')),
            requires_road_closure=bool(row.get('requires_road_closure', False)),
            start_dt=start_dt,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            latitude=float(row.get('latitude', 0.0)) if not pd.isna(row.get('latitude')) else 0.0,
            longitude=float(row.get('longitude', 0.0)) if not pd.isna(row.get('longitude')) else 0.0,
            end_latitude=float(row.get('endlatitude', 0.0)) if not pd.isna(row.get('endlatitude')) else None,
            end_longitude=float(row.get('endlongitude', 0.0)) if not pd.isna(row.get('endlongitude')) else None,
            corridor=str(row.get('corridor', 'Non-corridor')),
            zone=str(row.get('zone')) if not pd.isna(row.get('zone')) else None,
            junction=str(row.get('junction')) if not pd.isna(row.get('junction')) else None,
            veh_type=str(row.get('veh_type')) if not pd.isna(row.get('veh_type')) else None,
            route_path=row.get('parsed_route_path'),
            clearance_minutes=clearance_minutes,
            status=str(row.get('status', 'closed'))
        )
        events.append(event)
    return events

@st.cache_data
def load_clearance_dataset() -> pd.DataFrame:
    df = load_raw_data()
    if df.empty:
        return df
        
    # Apply filtering rules for clearance dataset
    if 'clearance_end' in df.columns:
        mask = df['clearance_end'].notna()
    else:
        return pd.DataFrame()
        
    if 'clearance_minutes' in df.columns:
        mask &= (df['clearance_minutes'] > 0)
        mask &= (df['clearance_minutes'] <= 2880)
        
    if 'status' in df.columns:
        mask &= (df['status'].str.lower() != 'active')
        
    return df[mask].copy()
