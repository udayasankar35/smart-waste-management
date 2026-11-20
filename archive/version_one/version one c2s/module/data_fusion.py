#!/usr/bin/env python3
"""
Data Fusion Module
AI-Enabled Smart Waste Segregation & Monitoring System

Combines YOLOv8 AI detections with Raspberry Pi Pico W sensor data.

Key behaviours:
  - strict_sensor_mode=True  → raises RuntimeError if Pico is not connected.
                                 NEVER produces simulated readings in production.
  - strict_sensor_mode=False → falls back to simulation, logs a PROMINENT WARNING
                                 on each simulated frame so the operator knows.

Fusion logic:
  - High humidity + moisture  → refine to wet_waste
  - Low moisture              → refine to dry_waste
  - High gas level            → override to hazardous
  - Ultrasonic distance       → bin fill percentage
"""

import json
import time
import logging
import serial
import serial.tools.list_ports
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from config import (
    SensorThresholds, SerialConfig, VehicleConfig,
    DEFAULT_CONFIG, WasteType
)
from waste_detector import WasteFrame, WasteDetection

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """Parsed sensor data from Raspberry Pi Pico W"""
    timestamp:   float = 0.0
    reading_id:  int   = 0

    # Ultrasonic (HC-SR04) — GP3/GP2
    distance_cm:  float = -1.0
    bin_fill_pct: float = -1.0

    # DHT11 Temperature & Humidity — GP15
    temperature_c: float = -1.0
    humidity_pct:  float = -1.0

    # MQ Gas Sensor — GP26 (analog), GP16 (digital)
    gas_analog:  int = 0
    gas_digital: int = 0

    # HW-103 Soil Moisture — GP27 (analog)
    moisture_raw: int = 0

    # Per-sensor health flags (from Pico firmware sensor_health block)
    ultrasonic_ok: bool = True
    dht11_ok:      bool = True
    gas_ok:        bool = True
    moisture_ok:   bool = True

    # Source tracking — CRITICAL for distinguishing real vs fake data
    is_valid:     bool = False
    is_simulated: bool = False   # True when no Pico is connected

    @staticmethod
    def from_json(data: dict) -> "SensorReading":
        """Parse JSON payload from Pico W firmware"""
        try:
            sensors = data.get("sensors", {})
            ultra   = sensors.get("ultrasonic", {})
            dht     = sensors.get("dht11",      {})
            gas     = sensors.get("gas",         {})
            moist   = sensors.get("moisture",    {})
            health  = data.get("sensor_health",  {})

            return SensorReading(
                timestamp    = time.time(),
                reading_id   = data.get("reading_id", 0),
                distance_cm  = ultra.get("distance_cm",  -1.0),
                bin_fill_pct = ultra.get("bin_fill_pct", -1.0),
                temperature_c = dht.get("temperature_c", -1.0),
                humidity_pct  = dht.get("humidity_pct",  -1.0),
                gas_analog    = gas.get("analog",  0),
                gas_digital   = gas.get("digital", 0),
                moisture_raw  = moist.get("raw",   0),
                ultrasonic_ok = health.get("ultrasonic", "ok") == "ok",
                dht11_ok      = health.get("dht11",      "ok") == "ok",
                gas_ok        = health.get("gas",        "ok") == "ok",
                moisture_ok   = health.get("moisture",   "ok") == "ok",
                is_valid      = True,
                is_simulated  = False,
            )
        except Exception as e:
            logger.error(f"❌ Sensor JSON parse error: {e}")
            return SensorReading(is_valid=False, is_simulated=False)


@dataclass
class WasteRecord:
    """Unified waste detection record (AI + sensors)"""
    timestamp:  str = ""
    waste_type: str = "unknown"
    ai_confidence: float = 0.0
    ai_class:   str = ""

    # Sensor values
    humidity:      float = -1.0
    temperature:   float = -1.0
    gas_level:     int   = 0
    moisture_level: int  = 0
    bin_fill_level: float = -1.0

    # Derived flags
    is_wet:       bool = False
    is_hazardous: bool = False
    gas_alert:    str  = "safe"
    bin_alert:    str  = "normal"

    # Data source tag — distinguishes real hardware readings from simulation
    data_source: str = "real"    # "real" | "simulated"

    # Vehicle
    vehicle_id: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp":      self.timestamp,
            "waste_type":     self.waste_type,
            "ai_confidence":  round(self.ai_confidence, 4),
            "ai_class":       self.ai_class,
            "humidity":       round(self.humidity, 1),
            "temperature":    round(self.temperature, 1),
            "gas_level":      self.gas_level,
            "moisture_level": self.moisture_level,
            "bin_fill_level": round(self.bin_fill_level, 1),
            "is_wet":         self.is_wet,
            "is_hazardous":   self.is_hazardous,
            "gas_alert":      self.gas_alert,
            "bin_alert":      self.bin_alert,
            "data_source":    self.data_source,
            "vehicle_id":     self.vehicle_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─── Serial Reader ──────────────────────────────────────────────────────────────

class PicoSerialReader:
    """
    Reads sensor data from Raspberry Pi Pico W over USB serial (UART).

    strict_sensor_mode=True:
        If the Pico is NOT detected on the serial port, connect() raises
        RuntimeError and the system halts. No simulated data, ever.

    strict_sensor_mode=False (default / dev mode):
        If the Pico is absent, falls back to simulated readings.
        Every simulated reading prints a WARNING so the operator knows.
    """

    def __init__(self, config: SerialConfig = None):
        self.config      = config or DEFAULT_CONFIG.serial
        self.serial_port: Optional[serial.Serial] = None
        self.last_reading: Optional[SensorReading] = None
        self.read_count:   int  = 0
        self.error_count:  int  = 0
        self._connected:   bool = False

    # ── Connection ──────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Open serial connection to Pico W.

        In strict mode: raises RuntimeError if Pico is not found.
        In normal mode: logs a warning and returns False (simulation fallback).
        """
        try:
            self.serial_port = serial.Serial(
                port     = self.config.port,
                baudrate = self.config.baudrate,
                timeout  = self.config.timeout,
            )
            self._connected = True
            logger.info(
                f"✅ [HARDWARE] Pico W connected: {self.config.port} "
                f"@ {self.config.baudrate} baud"
            )
            return True

        except serial.SerialException as e:
            self._connected = False
            msg = (
                f"Pico W not found on {self.config.port}: {e}\n"
                f"  Available ports: {self._list_ports()}"
            )
            if self.config.strict_sensor_mode:
                logger.critical(
                    f"❌ STRICT MODE: {msg}\n"
                    f"   Connect the Pico W and restart, or use --no-strict."
                )
                raise RuntimeError(
                    f"STRICT MODE: Pico W not connected on {self.config.port}. "
                    f"Refusing to start with simulated sensor data."
                )
            else:
                logger.warning(
                    f"⚠️  {msg}\n"
                    f"   Running in SIMULATION MODE — sensor data is FAKE.\n"
                    f"   Use --strict to enforce real hardware."
                )
                return False

    def reconnect(self) -> bool:
        """Attempt to re-open the serial port (called after serial errors)."""
        logger.info(f"🔄 Attempting serial reconnect on {self.config.port}...")
        self.close()
        time.sleep(2)
        try:
            return self.connect()
        except RuntimeError:
            raise  # Re-raise in strict mode

    def _list_ports(self) -> str:
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            return ", ".join(ports) if ports else "(none found)"
        except Exception:
            return "(unable to list)"

    # ── Reading ─────────────────────────────────────────────────────────────────

    def read_sensors(self) -> SensorReading:
        """
        Read one JSON line from serial and parse to SensorReading.

        Returns real reading (is_simulated=False) when Pico is connected.
        Returns simulated reading (is_simulated=True) when no hardware,
        after logging a prominent WARNING.
        """
        if self.serial_port and self.serial_port.is_open:
            try:
                raw = self.serial_port.readline()
                if not raw:
                    # Timeout — Pico may be slow; return last known or simulate
                    return self._handle_no_data()

                line = raw.decode("utf-8").strip()
                if not line:
                    return self._handle_no_data()

                data    = json.loads(line)
                reading = SensorReading.from_json(data)
                self.last_reading = reading
                self.read_count  += 1

                # Log unhealthy sensors
                self._log_sensor_health(reading)

                logger.debug(
                    f"📡 [REAL] #{reading.reading_id} | "
                    f"Bin:{reading.bin_fill_pct:.0f}% | "
                    f"Temp:{reading.temperature_c:.1f}°C | "
                    f"Hum:{reading.humidity_pct:.0f}% | "
                    f"Gas:{reading.gas_analog} | "
                    f"Moist:{reading.moisture_raw}"
                )
                return reading

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.error_count += 1
                logger.debug(f"Serial parse error #{self.error_count}: {e}")
                return self._handle_no_data()

            except serial.SerialException as e:
                self.error_count += 1
                logger.error(f"🔌 Serial read error: {e} — attempting reconnect")
                try:
                    self.reconnect()
                except RuntimeError:
                    raise
                return self._handle_no_data()

        # No serial port open — simulate or abort
        return self._get_simulated_or_abort()

    def _handle_no_data(self) -> SensorReading:
        """Return last known reading if fresh, otherwise simulate."""
        if self.last_reading and (time.time() - self.last_reading.timestamp) < 5.0:
            return self.last_reading
        return self._get_simulated_or_abort()

    def _get_simulated_or_abort(self) -> SensorReading:
        """In strict mode: raise. In dev mode: return simulated reading."""
        if self.config.strict_sensor_mode:
            raise RuntimeError(
                "STRICT MODE: Lost connection to Pico W — cannot continue without real sensor data."
            )
        return self._simulate_reading()

    def _simulate_reading(self) -> SensorReading:
        """
        Generate simulated sensor data.
        Always logs a WARNING so operators are never confused.
        """
        import random

        if self.config.log_simulated_data:
            logger.warning(
                "⚠️  [SIMULATED] Frame sensor data is FAKE — Pico W not connected. "
                "Connect hardware and restart for real readings."
            )

        reading = SensorReading(
            timestamp     = time.time(),
            reading_id    = self.read_count,
            distance_cm   = round(random.uniform(5.0, 28.0), 1),
            bin_fill_pct  = round(random.uniform(10.0, 90.0), 1),
            temperature_c = round(random.uniform(22.0, 35.0), 1),
            humidity_pct  = round(random.uniform(30.0, 85.0), 1),
            gas_analog    = random.randint(100, 900),
            gas_digital   = 0,
            moisture_raw  = random.randint(5_000, 55_000),
            is_valid      = True,
            is_simulated  = True,
        )
        self.last_reading = reading
        self.read_count  += 1
        return reading

    @staticmethod
    def _log_sensor_health(reading: SensorReading):
        """Warn if any individual sensor on the Pico reported an error."""
        if not reading.ultrasonic_ok:
            logger.warning("⚠️  HC-SR04 ultrasonic sensor reported error on Pico")
        if not reading.dht11_ok:
            logger.warning("⚠️  DHT11 sensor reported error on Pico — temp/humidity invalid")
        if not reading.gas_ok:
            logger.warning("⚠️  MQ gas sensor reported error on Pico")
        if not reading.moisture_ok:
            logger.warning("⚠️  HW-103 moisture sensor reported error on Pico")

    # ── Cleanup ─────────────────────────────────────────────────────────────────

    def close(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            logger.info("🔌 Serial connection closed")

    @property
    def is_connected(self) -> bool:
        return self._connected and self.serial_port is not None and self.serial_port.is_open


# ─── Data Fusion Engine ─────────────────────────────────────────────────────────

class DataFusionEngine:
    """
    Combines AI waste detections with live sensor readings.
    Applies environmental logic to refine waste classification:
      - Sensors say wet  → override AI dry_waste → wet_waste
      - Gas danger level → override any type    → hazardous
    """

    def __init__(
        self,
        sensor_thresholds: SensorThresholds = None,
        vehicle_config:    VehicleConfig     = None,
    ):
        self.thresholds = sensor_thresholds or DEFAULT_CONFIG.sensors
        self.vehicle    = vehicle_config    or DEFAULT_CONFIG.vehicle
        self.records:   List[WasteRecord]   = []

    def fuse(
        self,
        waste_frame:    WasteFrame,
        sensor_reading: SensorReading,
    ) -> List[WasteRecord]:
        """
        Fuse AI detection results with sensor data.

        Returns:
            List of WasteRecord objects tagged with data_source="real"|"simulated"
        """
        records = []
        now     = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Evaluate environmental conditions
        gas_alert        = self._evaluate_gas(sensor_reading.gas_analog)
        bin_alert        = self._evaluate_bin_level(sensor_reading.bin_fill_pct)
        is_env_wet       = self._is_wet_environment(
                               sensor_reading.humidity_pct,
                               sensor_reading.moisture_raw,
                           )
        is_hazardous     = gas_alert in ("warning", "danger")
        data_source      = "simulated" if sensor_reading.is_simulated else "real"

        if waste_frame.detections:
            for det in waste_frame.detections:
                waste_type = self._refine_waste_type(
                    det.class_name, is_env_wet, is_hazardous
                )
                record = WasteRecord(
                    timestamp      = now,
                    waste_type     = waste_type,
                    ai_confidence  = det.confidence,
                    ai_class       = det.class_name,
                    humidity       = sensor_reading.humidity_pct,
                    temperature    = sensor_reading.temperature_c,
                    gas_level      = sensor_reading.gas_analog,
                    moisture_level = sensor_reading.moisture_raw,
                    bin_fill_level = sensor_reading.bin_fill_pct,
                    is_wet         = is_env_wet,
                    is_hazardous   = is_hazardous,
                    gas_alert      = gas_alert,
                    bin_alert      = bin_alert,
                    data_source    = data_source,
                    vehicle_id     = self.vehicle.vehicle_id,
                )
                records.append(record)
        else:
            # No detections — still log sensor state
            record = WasteRecord(
                timestamp      = now,
                waste_type     = "no_detection",
                humidity       = sensor_reading.humidity_pct,
                temperature    = sensor_reading.temperature_c,
                gas_level      = sensor_reading.gas_analog,
                moisture_level = sensor_reading.moisture_raw,
                bin_fill_level = sensor_reading.bin_fill_pct,
                is_wet         = is_env_wet,
                is_hazardous   = is_hazardous,
                gas_alert      = gas_alert,
                bin_alert      = bin_alert,
                data_source    = data_source,
                vehicle_id     = self.vehicle.vehicle_id,
            )
            records.append(record)

        self.records.extend(records)
        return records

    # ── Refinement Logic ────────────────────────────────────────────────────────

    def _refine_waste_type(self, ai_class: str, is_wet: bool, is_hazardous: bool) -> str:
        """Refine AI classification using sensor context."""
        if is_hazardous:
            return WasteType.HAZARDOUS.value

        # Sensors say wet environment but AI said dry → trust sensor (more reliable for smell/moisture)
        if is_wet and ai_class == WasteType.DRY_WASTE.value:
            return WasteType.WET_WASTE.value

        return ai_class

    def _evaluate_gas(self, gas_analog: int) -> str:
        if gas_analog >= self.thresholds.gas_danger_level:
            return "danger"
        elif gas_analog >= self.thresholds.gas_warning_level:
            return "warning"
        return "safe"

    def _evaluate_bin_level(self, fill_pct: float) -> str:
        if fill_pct < 0:
            return "unknown"
        if fill_pct >= self.thresholds.bin_full_threshold:
            return "full"
        elif fill_pct >= self.thresholds.bin_full_threshold * 0.75:
            return "nearly_full"
        return "normal"

    def _is_wet_environment(self, humidity: float, moisture: int) -> bool:
        humid = humidity >= self.thresholds.wet_waste_humidity if humidity >= 0 else False
        moist = moisture >= self.thresholds.wet_waste_threshold
        return humid or moist

    # ── Statistics ──────────────────────────────────────────────────────────────

    def get_recent_records(self, n: int = 10) -> List[WasteRecord]:
        return self.records[-n:]

    def get_statistics(self) -> Dict:
        if not self.records:
            return {"total_records": 0}

        type_counts = {}
        for r in self.records:
            type_counts[r.waste_type] = type_counts.get(r.waste_type, 0) + 1

        hazardous  = sum(1 for r in self.records if r.is_hazardous)
        wet        = sum(1 for r in self.records if r.is_wet)
        simulated  = sum(1 for r in self.records if r.data_source == "simulated")

        return {
            "total_records":       len(self.records),
            "waste_distribution":  type_counts,
            "hazardous_detections": hazardous,
            "wet_detections":      wet,
            "simulated_records":   simulated,
            "real_records":        len(self.records) - simulated,
            "vehicle_id":          self.vehicle.vehicle_id,
        }
