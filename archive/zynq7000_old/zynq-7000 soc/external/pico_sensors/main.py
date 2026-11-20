"""
Raspberry Pi Pico W — Sensor Hub  (MicroPython)
AI-Enabled Smart Waste Segregation & Monitoring System

Reads all 4 sensors and sends structured JSON over USB serial to
the Jetson Nano every second.

HARDWARE CONNECTIONS (match config.py PicoPins exactly):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HC-SR04 Ultrasonic Distance Sensor:
    TRIG  → GP3   (Pin 5)
    ECHO  → GP2   (Pin 4)
    VCC   → 5V
    GND   → GND

  DHT11 Temperature & Humidity:
    DATA  → GP15  (Pin 20)
    VCC   → 3.3V
    GND   → GND

  MQ Gas Sensor (MQ-2 / MQ-135):
    AO    → GP26  (Pin 31, ADC0)  ← analog output
    DO    → GP16  (Pin 21)        ← digital threshold output
    VCC   → 5V
    GND   → GND

  HW-103 Soil Moisture Sensor:
    AO    → GP27  (Pin 32, ADC1)  ← analog output
    VCC   → 3.3V
    GND   → GND

LED behaviour:
  Solid ON  during reading
  OFF       between readings
  Fast blink (3×)  on sensor error

Upload with Thonny or: mpremote cp pico_sensors/main.py :main.py
"""

import machine
import utime
import ujson as json
import sys

# ─── DHT11 (MicroPython built-in) ────────────────────────────────────────────
try:
    import dht as _dht_mod
    _DHT_AVAILABLE = True
except ImportError:
    _dht_mod       = None
    _DHT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  PIN SETUP  (must match config.py → PicoPins)
# ═══════════════════════════════════════════════════════════════════════════════

# HC-SR04 Ultrasonic
TRIG_PIN = machine.Pin(3, machine.Pin.OUT)
ECHO_PIN = machine.Pin(2, machine.Pin.IN)

# DHT11
DHT_PIN  = machine.Pin(15)

# MQ Gas Sensor
GAS_ADC     = machine.ADC(machine.Pin(26))   # ADC0 — analog gas level
GAS_DIGITAL = machine.Pin(16, machine.Pin.IN) # Digital threshold pin

# HW-103 Soil Moisture
MOISTURE_ADC = machine.ADC(machine.Pin(27))  # ADC1 — analog moisture level

# On-board status LED
LED = machine.Pin("LED", machine.Pin.OUT)


# ═══════════════════════════════════════════════════════════════════════════════
#  LED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def led_error_blink(n: int = 3):
    """Blink LED rapidly n times to signal a sensor error."""
    for _ in range(n):
        LED.on()
        utime.sleep_ms(80)
        LED.off()
        utime.sleep_ms(80)


# ═══════════════════════════════════════════════════════════════════════════════
#  SENSOR READING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def read_ultrasonic():
    """
    Measure HC-SR04 distance in centimetres.
    Returns (distance_cm, bin_fill_pct, status_str).
    Returns (-1.0, -1.0, "timeout") on failure.
    MAX_DISTANCE_CM must match config.py SensorThresholds.bin_max_distance_cm
    """
    MAX_DISTANCE_CM = 30.0

    try:
        # 10 µs trigger pulse
        TRIG_PIN.low()
        utime.sleep_us(2)
        TRIG_PIN.high()
        utime.sleep_us(10)
        TRIG_PIN.low()

        # Wait for echo HIGH (30 ms timeout)
        t_out = utime.ticks_us() + 30_000
        while ECHO_PIN.value() == 0:
            pulse_start = utime.ticks_us()
            if utime.ticks_diff(utime.ticks_us(), t_out) > 0:
                return -1.0, -1.0, "timeout"

        # Wait for echo LOW
        t_out = utime.ticks_us() + 30_000
        while ECHO_PIN.value() == 1:
            pulse_end = utime.ticks_us()
            if utime.ticks_diff(utime.ticks_us(), t_out) > 0:
                return -1.0, -1.0, "timeout"

        duration_us = utime.ticks_diff(pulse_end, pulse_start)
        distance    = round((duration_us * 0.0343) / 2.0, 1)

        # Sanity check
        if distance < 0 or distance > 400:
            return -1.0, -1.0, "out_of_range"

        fill = round(max(0.0, min(100.0,
               ((MAX_DISTANCE_CM - distance) / MAX_DISTANCE_CM) * 100.0)), 1)
        return distance, fill, "ok"

    except Exception as e:
        return -1.0, -1.0, "error"


def read_dht11():
    """
    Read DHT11 temperature (°C) and humidity (%).
    Returns (temp, humidity, status_str).
    Returns (None, None, "unavailable") if module missing,
            (None, None, "error") on read failure.
    """
    if not _DHT_AVAILABLE:
        return None, None, "unavailable"

    try:
        sensor = _dht_mod.DHT11(DHT_PIN)
        sensor.measure()
        temp = sensor.temperature()
        hum  = sensor.humidity()

        # Sanity range check for DHT11
        if not (-10 <= temp <= 80) or not (0 <= hum <= 100):
            return None, None, "out_of_range"

        return float(temp), float(hum), "ok"

    except Exception as e:
        led_error_blink(2)
        return None, None, "error"


def read_gas_sensor():
    """
    Read MQ gas sensor.
    Returns (analog_uint16, digital_bit, status_str).
    """
    try:
        analog  = GAS_ADC.read_u16()
        digital = GAS_DIGITAL.value()
        return analog, digital, "ok"
    except Exception as e:
        return 0, 0, "error"


def read_moisture_sensor():
    """
    Read HW-103 soil moisture sensor.
    Returns (raw_uint16, status_str).
    Higher value = wetter material.
    """
    try:
        raw = MOISTURE_ADC.read_u16()
        return raw, "ok"
    except Exception as e:
        return 0, "error"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main sensor loop — emits one JSON line per second over USB serial."""

    # 2-second warm-up (DHT11 needs ~1 s after power-on)
    LED.on()
    utime.sleep(2)
    LED.off()

    reading_id = 0

    while True:
        try:
            LED.on()

            # ── Read all sensors ──────────────────────────────────────────────
            dist_cm, fill_pct, ultra_status  = read_ultrasonic()
            temp_c,  hum_pct,  dht_status    = read_dht11()
            gas_raw, gas_dig,  gas_status     = read_gas_sensor()
            moist_raw, moist_status           = read_moisture_sensor()

            # ── Build JSON payload ────────────────────────────────────────────
            payload = {
                "source":     "pico_w",
                "reading_id": reading_id,
                "timestamp_ms": utime.ticks_ms(),
                "sensor_health": {
                    "ultrasonic": ultra_status,
                    "dht11":      dht_status,
                    "gas":        gas_status,
                    "moisture":   moist_status,
                },
                "sensors": {
                    "ultrasonic": {
                        "distance_cm":  dist_cm,
                        "bin_fill_pct": fill_pct,
                    },
                    "dht11": {
                        "temperature_c": temp_c,   # null if sensor error
                        "humidity_pct":  hum_pct,  # null if sensor error
                    },
                    "gas": {
                        "analog":  gas_raw,
                        "digital": gas_dig,
                    },
                    "moisture": {
                        "raw": moist_raw,
                    },
                },
            }

            # ── Send JSON line over USB serial ────────────────────────────────
            print(json.dumps(payload))    # MicroPython print → USB CDC stdout

            LED.off()
            reading_id += 1
            utime.sleep(1)

        except KeyboardInterrupt:
            LED.off()
            break

        except Exception as e:
            led_error_blink(3)
            LED.off()
            utime.sleep(1)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
