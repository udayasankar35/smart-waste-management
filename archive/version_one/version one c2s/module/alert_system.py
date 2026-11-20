#!/usr/bin/env python3
"""
Alert System Module
AI-Enabled Smart Waste Segregation & Monitoring System

Threshold-based alert system for:
  - Toxic gas levels (MQ sensor)
  - Bin overflow (ultrasonic sensor)
  - Hazardous waste detection (AI)
  - Environmental hazards (humidity extremes)

Alert channels: console logs, email (SMTP), database logging.
"""

import time
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import AlertConfig, SensorThresholds, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ─── Alert Data ─────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    """Single alert event"""
    alert_type: str         # gas, bin_overflow, hazardous_waste, humidity
    severity: str           # info, warning, danger, critical
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    vehicle_id: str = ""
    data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "vehicle_id": self.vehicle_id,
            "data": self.data,
        }


# ─── Alert System ───────────────────────────────────────────────────────────────

class AlertSystem:
    """
    Monitors waste records and triggers alerts based on thresholds.
    Supports multiple notification channels with cooldown.
    """

    SEVERITY_ICONS = {
        "info": "ℹ️",
        "warning": "⚠️",
        "danger": "🚨",
        "critical": "🔴",
    }

    def __init__(
        self,
        alert_config: AlertConfig = None,
        sensor_thresholds: SensorThresholds = None,
    ):
        self.config = alert_config or DEFAULT_CONFIG.alerts
        self.thresholds = sensor_thresholds or DEFAULT_CONFIG.sensors
        self.alert_history: List[Alert] = []
        self._last_alert_times: Dict[str, float] = {}

        if not self.config.enabled:
            logger.info("Alert system disabled in config")

    def check_and_alert(self, record: Dict, db=None) -> List[Alert]:
        """
        Check a waste record against all thresholds and trigger alerts.

        Args:
            record: WasteRecord dict
            db: Optional WasteDatabase for logging alerts

        Returns:
            List of triggered Alert objects
        """
        if not self.config.enabled:
            return []

        triggered = []

        # ── Gas level check ──
        gas_level = record.get("gas_level", 0)
        if gas_level >= self.thresholds.gas_danger_level:
            alert = self._create_alert(
                "gas", "critical",
                f"TOXIC GAS DANGER! Level: {gas_level} (threshold: {self.thresholds.gas_danger_level})",
                record, {"gas_level": gas_level}
            )
            if alert:
                triggered.append(alert)

        elif gas_level >= self.thresholds.gas_warning_level:
            alert = self._create_alert(
                "gas", "warning",
                f"Gas level elevated: {gas_level} (threshold: {self.thresholds.gas_warning_level})",
                record, {"gas_level": gas_level}
            )
            if alert:
                triggered.append(alert)

        # ── Bin overflow check ──
        bin_level = record.get("bin_fill_level", -1)
        if bin_level >= self.thresholds.bin_full_threshold:
            alert = self._create_alert(
                "bin_overflow", "danger",
                f"BIN FULL! Fill level: {bin_level:.0f}% (threshold: {self.thresholds.bin_full_threshold}%)",
                record, {"bin_fill_level": bin_level}
            )
            if alert:
                triggered.append(alert)

        # ── Hazardous waste check ──
        waste_type = record.get("waste_type")
        if record.get("is_hazardous", False) and waste_type and waste_type not in ("no_detection", "unknown"):
            alert = self._create_alert(
                "hazardous_waste", "critical",
                f"HAZARDOUS WASTE detected! Type: {waste_type}",
                record, {"waste_type": waste_type}
            )
            if alert:
                triggered.append(alert)

        # ── Extreme humidity check ──
        humidity = record.get("humidity", -1)
        if humidity >= self.thresholds.high_humidity_alert:
            alert = self._create_alert(
                "humidity", "warning",
                f"High humidity: {humidity:.0f}% — environment may be hazardous",
                record, {"humidity": humidity}
            )
            if alert:
                triggered.append(alert)

        # Deliver alerts
        for alert in triggered:
            self._deliver(alert, db)

        return triggered

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        record: Dict,
        data: Dict
    ) -> Optional[Alert]:
        """Create alert with cooldown check"""
        # Check cooldown
        last_time = self._last_alert_times.get(alert_type, 0)
        if time.time() - last_time < self.config.alert_cooldown_sec:
            return None  # Cooldown active

        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            vehicle_id=record.get("vehicle_id", ""),
            data=data,
        )

        self._last_alert_times[alert_type] = time.time()
        self.alert_history.append(alert)
        return alert

    def _deliver(self, alert: Alert, db=None):
        """Deliver alert through all configured channels"""
        icon = self.SEVERITY_ICONS.get(alert.severity, "🔔")

        # Console logging (always)
        if self.config.console_alerts:
            logger.warning(f"ALERT [{alert.severity}]: {alert.message}")

        # Database logging
        if db:
            try:
                db.insert_alert(
                    alert_type=alert.alert_type,
                    severity=alert.severity,
                    message=alert.message,
                    vehicle_id=alert.vehicle_id,
                )
            except Exception as e:
                logger.error(f"DB alert log failed: {e}")

        # Email notification
        if self.config.email_enabled and alert.severity in ("danger", "critical"):
            self._send_email(alert)

    def _send_email(self, alert: Alert):
        """Send email alert via SMTP"""
        if not self.config.sender_email or not self.config.recipient_emails:
            logger.debug("Email not configured — skipping")
            return

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 Waste Alert: {alert.alert_type.upper()} — {alert.severity.upper()}"
            msg["From"] = self.config.sender_email
            msg["To"] = ", ".join(self.config.recipient_emails)

            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #e74c3c;">⚠️ Smart Waste Alert</h2>
                <table style="border-collapse: collapse;">
                    <tr><td><b>Type:</b></td><td>{alert.alert_type}</td></tr>
                    <tr><td><b>Severity:</b></td><td>{alert.severity.upper()}</td></tr>
                    <tr><td><b>Message:</b></td><td>{alert.message}</td></tr>
                    <tr><td><b>Vehicle:</b></td><td>{alert.vehicle_id}</td></tr>
                    <tr><td><b>Time:</b></td><td>{alert.timestamp}</td></tr>
                </table>
                <p style="color: #7f8c8d; margin-top: 20px;">
                    <i>AI-Enabled Smart Waste Segregation & Monitoring System</i>
                </p>
            </body>
            </html>
            """

            msg.attach(MIMEText(body, "html"))

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(self.config.sender_email, self.config.recipient_emails, msg.as_string())

            logger.info(f"📧 Alert email sent to {self.config.recipient_emails}")

        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def get_recent_alerts(self, n: int = 20) -> List[Dict]:
        """Get recent alert dictionaries"""
        return [a.to_dict() for a in self.alert_history[-n:]]

    def get_alert_summary(self) -> Dict:
        """Summary of all alerts"""
        type_counts = {}
        severity_counts = {}
        for alert in self.alert_history:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

        return {
            "total_alerts": len(self.alert_history),
            "by_type": type_counts,
            "by_severity": severity_counts,
        }
