#!/usr/bin/env python3
"""
Database Module
AI-Enabled Smart Waste Segregation & Monitoring System

SQLite database for storing waste detection records, sensor readings,
and operational data. Provides CRUD operations and query helpers.
"""

import sqlite3
import os
import time
import logging
from typing import List, Dict, Optional
from contextlib import contextmanager

from config import DatabaseConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class WasteDatabase:
    """SQLite database for waste monitoring data"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_CONFIG.database.db_path
        if self.db_path != ":memory:" and os.path.dirname(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info(f"✅ Database initialized: {self.db_path}")

    def _init_db(self):
        """Create tables if they don't exist"""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS waste_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                waste_type TEXT NOT NULL,
                ai_confidence REAL DEFAULT 0.0,
                ai_class TEXT DEFAULT '',
                humidity REAL DEFAULT -1.0,
                temperature REAL DEFAULT -1.0,
                gas_level INTEGER DEFAULT 0,
                moisture_level INTEGER DEFAULT 0,
                bin_fill_level REAL DEFAULT -1.0,
                is_wet INTEGER DEFAULT 0,
                is_hazardous INTEGER DEFAULT 0,
                gas_alert TEXT DEFAULT 'safe',
                bin_alert TEXT DEFAULT 'normal',
                vehicle_id TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                distance_cm REAL DEFAULT -1.0,
                bin_fill_pct REAL DEFAULT -1.0,
                temperature_c REAL DEFAULT -1.0,
                humidity_pct REAL DEFAULT -1.0,
                gas_analog INTEGER DEFAULT 0,
                gas_digital INTEGER DEFAULT 0,
                moisture_raw INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                vehicle_id TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS json_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                report_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Create indexes for frequent queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_waste_timestamp ON waste_records(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_waste_type ON waste_records(waste_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_log(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_json_reports_timestamp ON json_reports(timestamp)")

        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get the persistent database connection"""
        return self._conn

    @contextmanager
    def _connect(self):
        """Context manager for database operations (uses persistent connection)"""
        conn = self._get_conn()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    # ─── Waste Records CRUD ─────────────────────────────────────────────────────

    def insert_waste_record(self, record: Dict) -> int:
        """Insert a single waste detection record"""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO waste_records
                (timestamp, waste_type, ai_confidence, ai_class,
                 humidity, temperature, gas_level, moisture_level,
                 bin_fill_level, is_wet, is_hazardous, gas_alert,
                 bin_alert, vehicle_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp", ""),
                record.get("waste_type", "unknown"),
                record.get("ai_confidence", 0.0),
                record.get("ai_class", ""),
                record.get("humidity", -1.0),
                record.get("temperature", -1.0),
                record.get("gas_level", 0),
                record.get("moisture_level", 0),
                record.get("bin_fill_level", -1.0),
                int(record.get("is_wet", False)),
                int(record.get("is_hazardous", False)),
                record.get("gas_alert", "safe"),
                record.get("bin_alert", "normal"),
                record.get("vehicle_id", ""),
            ))
            conn.commit()
            return cursor.lastrowid

    def insert_many_waste_records(self, records: List[Dict]) -> int:
        """Batch insert waste records"""
        count = 0
        for record in records:
            self.insert_waste_record(record)
            count += 1
        return count

    def get_recent_records(self, limit: int = 50) -> List[Dict]:
        """Get most recent waste records"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM waste_records ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_records_by_type(self, waste_type: str, limit: int = 100) -> List[Dict]:
        """Get records filtered by waste type"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM waste_records WHERE waste_type = ? ORDER BY id DESC LIMIT ?",
                (waste_type, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_records_by_date_range(self, start: str, end: str) -> List[Dict]:
        """Get records within date range"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM waste_records WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                (start, end)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_waste_distribution(self) -> Dict[str, int]:
        """Get count of each waste type"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT waste_type, COUNT(*) as count FROM waste_records GROUP BY waste_type ORDER BY count DESC"
            ).fetchall()
            return {row["waste_type"]: row["count"] for row in rows}

    def get_total_records(self) -> int:
        """Get total number of waste records"""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM waste_records").fetchone()[0]

    # ─── Sensor Log ──────────────────────────────────────────────────────────────

    def insert_sensor_reading(self, reading: Dict) -> int:
        """Insert a sensor log entry"""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO sensor_log
                (timestamp, distance_cm, bin_fill_pct, temperature_c,
                 humidity_pct, gas_analog, gas_digital, moisture_raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reading.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
                reading.get("distance_cm", -1.0),
                reading.get("bin_fill_pct", -1.0),
                reading.get("temperature_c", -1.0),
                reading.get("humidity_pct", -1.0),
                reading.get("gas_analog", 0),
                reading.get("gas_digital", 0),
                reading.get("moisture_raw", 0),
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_sensor_readings(self, limit: int = 100) -> List[Dict]:
        """Get recent sensor readings"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sensor_log ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    # ─── Alerts ──────────────────────────────────────────────────────────────────

    def insert_alert(self, alert_type: str, severity: str, message: str, vehicle_id: str = "") -> int:
        """Insert an alert record"""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO alerts (timestamp, alert_type, severity, message, vehicle_id)
                VALUES (?, ?, ?, ?, ?)
            """, (time.strftime("%Y-%m-%dT%H:%M:%S"), alert_type, severity, message, vehicle_id))
            conn.commit()
            return cursor.lastrowid

    # ─── JSON Report Storage ────────────────────────────────────────────────────

    def insert_json_report(self, report_type: str, payload: Dict) -> int:
        """Store a JSON report payload in the database"""
        import json

        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO json_reports (timestamp, report_type, payload)
                VALUES (?, ?, ?)
            """, (time.strftime("%Y-%m-%dT%H:%M:%S"), report_type, json.dumps(payload, default=str)))
            conn.commit()
            return cursor.lastrowid

    def get_recent_json_reports(self, limit: int = 20) -> List[Dict]:
        """Get recent stored JSON reports"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM json_reports ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_unresolved_alerts(self) -> List[Dict]:
        """Get all unresolved alerts"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE resolved = 0 ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    # ─── Statistics ──────────────────────────────────────────────────────────────

    def get_daily_statistics(self, date: str = None) -> Dict:
        """Get statistics for a given date (default today)"""
        if date is None:
            date = time.strftime("%Y-%m-%d")

        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM waste_records WHERE timestamp LIKE ?",
                (f"{date}%",)
            ).fetchone()[0]

            distribution = {}
            rows = conn.execute(
                "SELECT waste_type, COUNT(*) as count FROM waste_records WHERE timestamp LIKE ? GROUP BY waste_type",
                (f"{date}%",)
            ).fetchall()
            for row in rows:
                distribution[row["waste_type"]] = row["count"]

            hazardous = conn.execute(
                "SELECT COUNT(*) FROM waste_records WHERE timestamp LIKE ? AND is_hazardous = 1",
                (f"{date}%",)
            ).fetchone()[0]

            alerts_count = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?",
                (f"{date}%",)
            ).fetchone()[0]

        return {
            "date": date,
            "total_detections": total,
            "waste_distribution": distribution,
            "hazardous_count": hazardous,
            "alerts_count": alerts_count,
        }


# ─── Standalone Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🗄️  DATABASE TEST")
    print("=" * 40)

    db = WasteDatabase(":memory:")

    # Insert test records
    for wtype in ["plastic", "metal", "wet_waste", "dry_waste", "recyclable"]:
        db.insert_waste_record({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "waste_type": wtype,
            "ai_confidence": 0.87,
            "humidity": 55.0,
            "gas_level": 250,
            "bin_fill_level": 45.0,
            "vehicle_id": "GV-12",
        })

    print(f"Total records: {db.get_total_records()}")
    print(f"Distribution: {db.get_waste_distribution()}")
    print(f"Daily stats: {db.get_daily_statistics()}")
    print("✅ Database OK")
