"""
Tests for the 7-class model (material7-v1) support in waste_detector.py.

No weights, camera or GPU are needed: a fake YOLO object is used.

Run from the module folder:
    python -m pytest test_v2_taxonomy.py -q
or
    python test_v2_taxonomy.py
"""
import unittest

import numpy as np

from config import DetectionConfig
from waste_detector import WasteDetector, WasteFrame, WasteDetection


# ─── tiny fakes that look like Ultralytics results ───────────────────────────

class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, i):
        return _T(self.a[i])

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = _T([cls_id])
        self.conf = _T([conf])
        self.xyxy = _T([xyxy])


class _Result:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class FakeYolo:
    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes

    def predict(self, **kwargs):
        return [_Result(self.names, [_Box(*b) for b in self._boxes])]


V2_NAMES = {0: "plastic", 1: "glass", 2: "metal", 3: "paper_cardboard",
            4: "organic_waste", 5: "textile", 6: "e_waste"}
LEGACY_NAMES = {0: "plastic_bottle", 1: "glass_bottle", 2: "metal_can",
                3: "paper_cardboard", 4: "organic_waste"}
COCO_NAMES = {0: "cup", 1: "bottle"}

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def make_detector(names, custom_boxes, general_boxes=None):
    det = WasteDetector(DetectionConfig())
    det.model = FakeYolo(names, custom_boxes)
    det.is_custom_model = True
    det.general_model = FakeYolo(COCO_NAMES, general_boxes) if general_boxes else None
    det._detect_taxonomy()
    return det


class TestMapping(unittest.TestCase):
    def test_new_classes_use_existing_categories(self):
        m = WasteDetector.CUSTOM_MATERIAL_MAPPING
        self.assertEqual(m["textile"], "dry_waste")
        self.assertEqual(m["e_waste"], "hazardous")
        self.assertEqual(m["paper_cardboard"], "paper")
        self.assertEqual(m["organic_waste"], "organic")

    def test_normalize_new_names(self):
        n = WasteDetector.normalize_material_name
        self.assertEqual(n("textile"), "dry_waste")
        self.assertEqual(n("e_waste"), "hazardous")
        self.assertEqual(n("E-Waste"), "hazardous")
        self.assertEqual(n("organic_waste"), "organic")

    def test_old_behaviour_kept(self):
        n = WasteDetector.normalize_material_name
        self.assertEqual(n("milk jug"), "plastic")
        self.assertEqual(n("cardboard_box"), "cardboard")
        self.assertEqual(n("aluminum_can"), "metal")
        self.assertEqual(n("random object"), "unknown")


class TestTaxonomyDetection(unittest.TestCase):
    def test_v2_detected(self):
        det = make_detector(V2_NAMES, [])
        self.assertEqual(det.taxonomy, "material7-v1")
        self.assertEqual(det._plastic_bonus, 0.0)
        self.assertEqual(len(det.custom_class_names), 7)

    def test_legacy_detected(self):
        det = make_detector(LEGACY_NAMES, [])
        self.assertEqual(det.taxonomy, "legacy")
        self.assertEqual(det._plastic_bonus, WasteDetector.LEGACY_PLASTIC_BONUS)

    def test_statistics_report_taxonomy(self):
        det = make_detector(V2_NAMES, [])
        st = det.get_statistics()
        self.assertEqual(st["taxonomy"], "material7-v1")
        self.assertIn("e_waste", st["classes"])


class TestV2Detections(unittest.TestCase):
    def test_labels_and_categories(self):
        boxes = [
            (0, 0.90, (10, 10, 100, 100)),     # plastic
            (5, 0.80, (200, 10, 300, 100)),    # textile
            (6, 0.70, (400, 10, 500, 100)),    # e_waste
            (4, 0.60, (10, 200, 100, 300)),    # organic_waste
            (1, 0.60, (200, 200, 300, 300)),   # glass
        ]
        det = make_detector(V2_NAMES, boxes)
        wf = det.detect(FRAME, 1)
        got = {d.class_name: d.detected_object for d in wf.detections}
        self.assertEqual(got["plastic"], "Plastic")              # NOT "Plastic Bottle"
        self.assertEqual(got["dry_waste"], "Clothing / Fabric Textile")
        self.assertEqual(got["hazardous"], "Electronic Waste (E-Waste)")
        self.assertEqual(got["organic"], "Organic")
        self.assertEqual(got["glass"], "Glass")
        self.assertEqual(len(wf.detections), 5)

    def test_confident_custom_paper_box_not_overruled_by_weak_coco_cup(self):
        custom = [(3, 0.70, (50, 50, 250, 250))]          # paper_cardboard 0.70
        coco = [(0, 0.45, (60, 60, 240, 240))]            # COCO cup -> plastic 0.45
        det = make_detector(V2_NAMES, custom, coco)
        wf = det.detect(FRAME, 1)
        self.assertEqual([d.class_name for d in wf.detections], ["paper"])

    def test_weak_custom_paper_still_overruled_by_plastic(self):
        custom = [(3, 0.40, (50, 50, 250, 250))]          # weak paper 0.40
        coco = [(0, 0.60, (60, 60, 240, 240))]            # COCO cup -> plastic 0.60
        det = make_detector(V2_NAMES, custom, coco)
        wf = det.detect(FRAME, 1)
        self.assertEqual([d.class_name for d in wf.detections], ["plastic"])


class TestLegacyUnchanged(unittest.TestCase):
    def test_old_model_still_says_plastic_bottle(self):
        det = make_detector(LEGACY_NAMES, [(0, 0.80, (10, 10, 100, 100))])
        wf = det.detect(FRAME, 1)
        self.assertEqual(wf.detections[0].class_name, "plastic")
        self.assertEqual(wf.detections[0].detected_object, "Plastic Bottle")

    def test_old_model_plastic_bias_unchanged(self):
        custom = [(3, 0.70, (50, 50, 250, 250))]
        coco = [(0, 0.45, (60, 60, 240, 240))]
        det = make_detector(LEGACY_NAMES, custom, coco)
        wf = det.detect(FRAME, 1)
        # old rule: plastic overrules paper unless paper is > 0.85 confident
        self.assertEqual([d.class_name for d in wf.detections], ["plastic"])


class TestDrawing(unittest.TestCase):
    def test_hud_draws_new_categories(self):
        det = make_detector(V2_NAMES, [])
        wf = WasteFrame(frame_id=1, timestamp=0.0, image_shape=FRAME.shape)
        for i, (cls, obj) in enumerate([("dry_waste", "Clothing / Fabric Textile"),
                                        ("hazardous", "Electronic Waste (E-Waste)")]):
            x = 20 + i * 200
            wf.detections.append(WasteDetection(
                class_id=i, class_name=cls, confidence=0.8,
                bbox=(x, 60, x + 150, 200), centroid=(x + 75, 130),
                area=150 * 140, detected_object=obj))
        out = det.draw_detections(FRAME.copy(), wf)
        self.assertEqual(out.shape, FRAME.shape)


if __name__ == "__main__":
    unittest.main()
