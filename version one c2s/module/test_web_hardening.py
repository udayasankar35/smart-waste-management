"""
Tests for the web_app.py hardening patch.

Safe to run on your PC: it never touches your real database, advisor, Llama
model, YOLO weights or camera. Fake versions are used instead.

Run from the module folder:
    python -m pytest test_web_hardening.py -q
or
    python test_web_hardening.py
"""
import base64
import io
import json
import sys
import threading
import time
import types
import unittest

import cv2
import numpy as np


# ─── fakes (installed only while web_app is imported, then removed) ──────────────

class FakeDB:
    def __init__(self, path=None):
        self.rows = []

    def insert_waste_record(self, rec):
        self.rows.append(rec)

    def get_total_records(self):
        return len(self.rows)

    def get_waste_distribution(self):
        return {}

    def get_daily_statistics(self):
        return {}

    def get_recent_alerts(self, n=10):
        return []

    def get_recent_records(self, n=50):
        return self.rows[-n:]

    def get_records_by_type(self, t, n=50):
        return []

    def get_recent_sensor_readings(self, n=50):
        return []


class FakeAdvisor:
    def __init__(self, llama_engine=None):
        self.last_detection = None

    def guess_object_and_analysis(self, cn, conf, bbox, detected_object="", image=None):
        return {"guessed_object": detected_object or cn, "plastic_type": None,
                "recycling_steps": ["step"], "household_reuse": ["reuse"],
                "decomposition_time": "?"}

    def get_detection_advice(self, cn, conf, bbox, detected_object="", image=None):
        return {"message": "advice for " + cn}

    def set_last_detection(self, d):
        self.last_detection = d

    def get_response(self, m, detection_context=None):
        return {"message": "hi", "type": "greeting"}


class FakeLlama:
    load_status = "fake"
    active_backend = "fallback"

    def __init__(self, config=None):
        pass

    def initialize(self):
        pass

    def get_info(self):
        return {"status": "fake"}


def _import_web_app():
    fakes = {
        "database": types.SimpleNamespace(WasteDatabase=FakeDB),
        "waste_advisor": types.SimpleNamespace(WasteAdvisor=FakeAdvisor),
        "llama_engine": types.SimpleNamespace(LlamaEngine=FakeLlama),
    }
    saved = {k: sys.modules.get(k) for k in fakes}
    for k, v in fakes.items():
        mod = types.ModuleType(k)
        mod.__dict__.update(vars(v))
        sys.modules[k] = mod
    try:
        import web_app as wa
    finally:
        for k, old in saved.items():
            if old is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = old
    return wa


wa = _import_web_app()

from waste_detector import WasteDetector          # noqa: E402
from config import DetectionConfig                # noqa: E402
from test_v2_taxonomy import FakeYolo, V2_NAMES   # noqa: E402


# ─── a fake camera ───────────────────────────────────────────────────────────────

class FakeCap:
    opened = 0            # how many times the camera was opened
    live = 0              # how many are open right now
    max_live = 0

    def __init__(self):
        FakeCap.opened += 1
        FakeCap.live += 1
        FakeCap.max_live = max(FakeCap.max_live, FakeCap.live)
        self._n = 0
        self.closed = False

    @classmethod
    def reset(cls):
        cls.opened = cls.live = cls.max_live = 0

    def isOpened(self):
        return not self.closed

    def set(self, *a):
        return True

    def read(self):
        time.sleep(0.005)
        self._n += 1
        frame = np.full((120, 160, 3), self._n % 255, dtype=np.uint8)
        return True, frame

    def release(self):
        if not self.closed:
            self.closed = True
            FakeCap.live -= 1


def _jpeg(w=160, h=120):
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 90, dtype=np.uint8))
    assert ok
    return buf.tobytes()


class Base(unittest.TestCase):
    def setUp(self):
        det = WasteDetector(DetectionConfig())
        det.model = FakeYolo(V2_NAMES, [(0, 0.9, (10, 10, 90, 90))])   # one plastic
        det.is_custom_model = True
        det.general_model = None
        det._detect_taxonomy()
        self.det = det

        self.calls = 0
        self.fids = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        real_detect = det.detect

        def counting_detect(frame, frame_id=0):
            with self._lock:
                self.calls += 1
                self.fids.append(frame_id)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)                 # make overlaps likely if there is no lock
                return real_detect(frame, frame_id)
            finally:
                with self._lock:
                    self.active -= 1

        det.detect = counting_detect

        wa.detector = det
        wa.db = FakeDB()
        wa.advisor = FakeAdvisor()
        wa.llama_engine = FakeLlama()
        wa.frame_counter = 0
        wa._stream_cache.update(fid=-1, jpeg=None)

        FakeCap.reset()
        self._orig_open = wa._open_camera
        wa._open_camera = lambda idx=0: FakeCap()
        wa._camera_hub.IDLE_RELEASE_SEC = 0.4
        self.client = wa.app.test_client()

    def tearDown(self):
        wa._open_camera = self._orig_open
        # let the shared camera thread finish before the next test
        deadline = time.time() + 3
        while wa._camera_hub._thread is not None and time.time() < deadline:
            time.sleep(0.05)


class TestUploads(Base):
    def test_valid_upload_uses_new_plastic_label(self):
        r = self.client.post("/api/detect", data={"image": (io.BytesIO(_jpeg()), "a.jpg")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual(j["num_detections"], 1)
        self.assertEqual(j["detections"][0]["class_name"], "plastic")
        self.assertEqual(j["detections"][0]["detected_object"], "Plastic")
        self.assertEqual(len(wa.db.rows), 1)

    def test_base64_json_upload(self):
        b64 = "data:image/jpeg;base64," + base64.b64encode(_jpeg()).decode()
        r = self.client.post("/api/detect", data=json.dumps({"image": b64}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 200)

    def test_oversized_upload_gets_413_not_500(self):
        big = b"0" * (26 * 1024 * 1024)
        r = self.client.post("/api/detect", data={"image": (io.BytesIO(big), "big.jpg")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 413)
        self.assertIn("too large", r.get_json()["error"].lower())

    def test_bad_base64_is_400(self):
        r = self.client.post("/api/detect", data=json.dumps({"image": "%%%not-base64%%%"}),
                             content_type="application/json")
        self.assertIn(r.status_code, (400,))

    def test_json_that_is_not_an_object_is_400(self):
        r = self.client.post("/api/detect", data=json.dumps([1, 2, 3]),
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_garbage_bytes_are_400(self):
        r = self.client.post("/api/detect", data={"image": (io.BytesIO(b"not an image"), "x.jpg")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_parallel_uploads_never_run_two_inferences_at_once(self):
        body = _jpeg()
        codes = []

        def work():
            c = wa.app.test_client()
            r = c.post("/api/detect", data={"image": (io.BytesIO(body), "a.jpg")},
                       content_type="multipart/form-data")
            codes.append(r.status_code)

        threads = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(codes, [200] * 8)
        self.assertEqual(self.max_active, 1)


class TestCamera(Base):
    def test_many_viewers_share_one_camera(self):
        errors = []

        def viewer():
            try:
                gen = wa.generate_frames()
                for _ in range(6):
                    chunk = next(gen)
                    assert chunk.startswith(b"--frame")
                gen.close()
            except Exception as exc:                      # pragma: no cover
                errors.append(repr(exc))

        results = {}

        def capture():
            c = wa.app.test_client()
            results["capture"] = c.post("/api/webcam/capture")

        def status():
            c = wa.app.test_client()
            results["status"] = c.get("/api/webcam/status")

        threads = [threading.Thread(target=viewer) for _ in range(3)]
        threads += [threading.Thread(target=capture), threading.Thread(target=status)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        self.assertEqual(errors, [])
        self.assertEqual(FakeCap.opened, 1, "camera must be opened exactly once")
        self.assertEqual(FakeCap.max_live, 1)
        self.assertEqual(results["capture"].status_code, 200)
        self.assertEqual(results["capture"].get_json()["detections"][0]["detected_object"], "Plastic")
        self.assertTrue(results["status"].get_json()["available"])
        self.assertEqual(self.max_active, 1, "inference must never overlap")

    def test_camera_released_after_last_viewer_and_reopened(self):
        r = self.client.get("/api/webcam/status")
        self.assertTrue(r.get_json()["available"])
        deadline = time.time() + 4
        while FakeCap.live and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(FakeCap.live, 0, "camera should be released when nobody watches")
        r = self.client.get("/api/webcam/status")
        self.assertTrue(r.get_json()["available"])
        self.assertEqual(FakeCap.opened, 2)

    def test_stream_detects_every_third_frame_once_for_all_viewers(self):
        gens = [wa.generate_frames() for _ in range(3)]
        for _ in range(12):
            for g in gens:
                next(g)
        [g.close() for g in gens]
        self.assertGreater(self.calls, 0)
        self.assertTrue(all(f % 3 == 0 for f in self.fids), self.fids)
        self.assertEqual(len(self.fids), len(set(self.fids)),
                         "three viewers must not triple the inference work")

    def test_same_frame_is_processed_once_for_concurrent_viewers(self):
        frame = np.full((120, 160, 3), 77, dtype=np.uint8)
        out = []

        def viewer():
            out.append(wa._stream_jpeg(frame, 3))      # frame id 3 -> a detection frame

        threads = [threading.Thread(target=viewer) for _ in range(5)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(self.calls, 1, "five viewers of the same frame = one inference")
        self.assertEqual(len(set(out)), 1, "all viewers get identical JPEG bytes")
        self.assertTrue(out[0] and out[0][:2] == b"\xff\xd8")

    def test_no_camera(self):
        wa._open_camera = lambda idx=0: None
        r = self.client.post("/api/webcam/capture")
        self.assertEqual(r.status_code, 503)
        r = self.client.get("/api/webcam/status")
        self.assertFalse(r.get_json()["available"])
        gen = wa.generate_frames()
        self.assertTrue(next(gen).startswith(b"--frame"))      # placeholder frame
        gen.close()


class TestMisc(Base):
    def test_model_info_reports_taxonomy(self):
        j = self.client.get("/api/model/info").get_json()
        self.assertTrue(j["success"])
        self.assertEqual(j["taxonomy"], "material7-v1")
        self.assertIn("e_waste", j["classes"])

    def test_wrong_model_path_stops_the_app(self):
        with self.assertRaises(SystemExit):
            wa.init_components("definitely/not/here.pt")

    def test_default_host_is_localhost(self):
        src = open(wa.__file__, encoding="utf-8").read()
        self.assertIn('default="127.0.0.1"', src)


if __name__ == "__main__":
    unittest.main()
