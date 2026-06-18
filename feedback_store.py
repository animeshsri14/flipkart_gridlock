import sqlite3
import os
import streamlit as st
import pandas as pd

DB_FILE = os.path.join(os.path.dirname(__file__), "gridlock_feedback.db")

@st.cache_resource
def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    # Initialize the schema
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id      TEXT NOT NULL,
            corridor      TEXT NOT NULL,
            event_cause   TEXT NOT NULL,
            priority      TEXT NOT NULL,
            hour_bucket   TEXT NOT NULL,
            event_type    TEXT NOT NULL,        -- 'planned' | 'unplanned'
            actual_clearance_min  REAL NOT NULL,
            actual_manpower       INTEGER,
            actual_barricades     INTEGER,
            submitted_at  TEXT NOT NULL DEFAULT (datetime('now')),
            notes         TEXT
        );
    ''')
    conn.commit()
    return conn

def save_feedback(
    event_id: str,
    corridor: str,
    event_cause: str,
    priority: str,
    hour_bucket: str,
    event_type: str,
    actual_clearance_min: float,
    actual_manpower: int = None,
    actual_barricades: int = None,
    notes: str = None
):
    """
    Persists feedback to SQLite and invalidates the cache so the forecaster
    re-queries the augmented dataset.
    """
    conn = get_connection()
    conn.execute('''
        INSERT INTO feedback (
            event_id, corridor, event_cause, priority, hour_bucket, event_type, 
            actual_clearance_min, actual_manpower, actual_barricades, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        event_id, corridor, event_cause, priority, hour_bucket, event_type,
        actual_clearance_min, actual_manpower, actual_barricades, notes
    ))
    conn.commit()
    
    # Invalidate cache so that forecaster re-queries the augmented dataset
    st.cache_data.clear()

def get_feedback() -> pd.DataFrame:
    """
    Retrieves all feedback as a pandas DataFrame.
    """
    conn = get_connection()
    return pd.read_sql_query("SELECT * FROM feedback", conn)
