"""Persistent SQLite Relational Storage & CSV Event Logging Handler."""

import os
import sqlite3
import csv
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import cv2
import numpy as np
import logging

logger = logging.getLogger("SecuritySystem.Logger")

class SecurityLogger:
    """Enterprise event logger managing SQLite relational tables and CSV backups."""

    def __init__(self, db_path: str = "database/security.db", csv_path: str = "logs/security_events.csv", snapshots_dir: str = "database/snapshots"):
        self.db_path = db_path
        self.csv_path = csv_path
        self.snapshots_dir = snapshots_dir

        self._ensure_directories()
        self._init_sqlite()
        self._init_csv()

    def _ensure_directories(self):
        """Creates target directory paths if they do not exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        os.makedirs(os.path.abspath(self.snapshots_dir), exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database with timeout protection."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        """Initializes database schema and indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    zone_name TEXT,
                    zone_id TEXT,
                    track_id INTEGER,
                    duration REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    snapshot_path TEXT,
                    message TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON security_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON security_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_zone ON security_events(zone_name)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    fps REAL,
                    active_targets INTEGER,
                    motion_pct REAL
                )
            """)
            conn.commit()

    def _init_csv(self):
        """Ensures the backup CSV file has proper header structure."""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "event_type", "zone_name", "zone_id",
                    "track_id", "duration", "confidence", "snapshot_path", "message"
                ])

    def log_event(self, event_data: Dict[str, Any], frame: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Logs a security incident to SQLite and CSV. Saves snapshot if frame is provided.

        Returns:
            Snapshot file path if saved, else None.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_type = event_data.get("event_type", "ALERT")
        zone_name = event_data.get("zone_name", "Unknown Zone")
        zone_id = event_data.get("zone_id", "")
        track_id = event_data.get("track_id", -1)
        duration = float(event_data.get("duration", 0.0))
        confidence = float(event_data.get("confidence", 0.0))
        message = event_data.get("message", "")

        snapshot_path = ""
        if frame is not None:
            filename = f"snap_{int(time.time())}_id{track_id}_{event_type.lower()}.jpg"
            full_snap_path = os.path.join(self.snapshots_dir, filename)
            try:
                cv2.imwrite(full_snap_path, frame)
                snapshot_path = full_snap_path
            except Exception as e:
                logger.error(f"Failed to save snapshot: {e}")

        # Insert to SQLite
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO security_events (
                        timestamp, event_type, zone_name, zone_id, track_id, duration, confidence, snapshot_path, message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, event_type, zone_name, zone_id, track_id, duration, confidence, snapshot_path, message))
                conn.commit()
        except Exception as e:
            logger.error(f"SQLite insertion error: {e}")

        # Append to CSV Archive
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, event_type, zone_name, zone_id, track_id, duration, confidence, snapshot_path, message
                ])
        except Exception as e:
            logger.error(f"CSV logging error: {e}")

        return snapshot_path if snapshot_path else None

    def query_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        zone_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """Queries security events with dynamic filtering and returns a pandas DataFrame."""
        query = "SELECT * FROM security_events WHERE 1=1"
        params = []

        if event_type and event_type != "All":
            query += " AND event_type = ?"
            params.append(event_type)

        if zone_name and zone_name != "All":
            query += " AND zone_name = ?"
            params.append(zone_name)

        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)

        if date_to:
            query += " AND timestamp <= ?"
            params.append(date_to)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        try:
            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
                return df
        except Exception as e:
            logger.error(f"Query error: {e}")
            return pd.DataFrame()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates high-level forensic KPI metrics for dashboard display."""
        stats = {
            "total_incidents": 0,
            "intrusions": 0,
            "loitering": 0,
            "unique_targets": 0,
            "last_event_time": "None"
        }
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM security_events")
                stats["total_incidents"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM security_events WHERE event_type = 'INTRUSION'")
                stats["intrusions"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM security_events WHERE event_type = 'LOITERING'")
                stats["loitering"] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT track_id) FROM security_events WHERE track_id > 0")
                stats["unique_targets"] = cursor.fetchone()[0]

                cursor.execute("SELECT timestamp FROM security_events ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    stats["last_event_time"] = row[0]
        except Exception as e:
            logger.error(f"Summary query error: {e}")

        return stats

    def clear_database(self):
        """Clears SQLite events table and truncates the CSV file."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM security_events")
            conn.execute("DELETE FROM telemetry_logs")
            conn.commit()
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "event_type", "zone_name", "zone_id",
                "track_id", "duration", "confidence", "snapshot_path", "message"
            ])
