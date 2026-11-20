import unittest
from waste_detector import WasteDetector

class TestMaterialMapping(unittest.TestCase):
    def test_material_mapping_for_common_waste_items(self):
        self.assertEqual(WasteDetector.normalize_material_name("milk jug"), "plastic")
        self.assertEqual(WasteDetector.normalize_material_name("plastic bottle"), "plastic")

    def test_unknown_item_stays_safe(self):
        self.assertEqual(WasteDetector.normalize_material_name("random object"), "unknown")

if __name__ == "__main__":
    unittest.main()
