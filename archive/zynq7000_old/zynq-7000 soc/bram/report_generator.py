#!/usr/bin/env python3
"""
Report Generator Module
AI-Enabled Smart Waste Segregation & Monitoring System

Generates structured reports from waste detection data:
    - JSON reports (structured data export)
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from config import ReportConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates JSON report payloads from waste detection data"""

    def __init__(self, config: ReportConfig = None):
        self.config = config or DEFAULT_CONFIG.reports

    # ─── JSON Report ────────────────────────────────────────────────────────────

    def generate_json(self, records: List[Dict], metadata: Dict = None) -> Dict:
        """
        Generate a JSON report payload.

        Args:
            records: List of waste record dictionaries
            metadata: Optional metadata dict

        Returns:
            JSON report dictionary
        """
        report = {
            "report_type": "waste_monitoring",
            "generated_at": datetime.now().isoformat(),
            "total_records": len(records),
            "metadata": metadata or {},
            "records": records
        }

        logger.info(f"📄 JSON report payload prepared ({len(records)} records)")
        return report

    # ─── Generate All Reports ───────────────────────────────────────────────────

    def generate_all(self, records: List[Dict], daily_stats: Dict = None) -> Dict[str, Dict]:
        """Generate all enabled report payloads"""
        payloads = {}

        if self.config.json_enabled:
            payloads["json"] = self.generate_json(records, {"daily_stats": daily_stats})

        logger.info(f"📦 Generated {len(payloads)} report payloads")
        return payloads


# ─── Standalone Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n📄 REPORT GENERATOR TEST")
    print("=" * 40)

    # Create sample records
    sample_records = []
    for i, wtype in enumerate(["plastic", "metal", "wet_waste", "dry_waste", "recyclable"] * 3):
        sample_records.append({
            "timestamp": f"2026-03-12T14:{i:02d}:00",
            "waste_type": wtype,
            "ai_confidence": 0.85 + (i % 5) * 0.02,
            "humidity": 55 + i,
            "temperature": 28.0,
            "gas_level": 200 + i * 30,
            "moisture_level": 15000 + i * 2000,
            "bin_fill_level": 40 + i * 3,
            "is_wet": wtype == "wet_waste",
            "is_hazardous": False,
            "gas_alert": "safe",
            "bin_alert": "normal",
            "vehicle_id": "GV-12",
        })

    daily_stats = {
        "date": "2026-03-12",
        "total_detections": len(sample_records),
        "waste_distribution": {"plastic": 3, "metal": 3, "wet_waste": 3, "dry_waste": 3, "recyclable": 3},
        "hazardous_count": 0,
        "alerts_count": 1,
    }

    gen = ReportGenerator()
    paths = gen.generate_all(sample_records, daily_stats)
    print(f"\n✅ Reports generated: {paths}")
