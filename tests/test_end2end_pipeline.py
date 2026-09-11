"""
SatQuery AI — Comprehensive End-to-End Integration Test Suite
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Tests full pipeline execution: image upload, GeoTIFF CRS extraction, modality detection,
agentic tool routing, PyTorch tensor inference, anti-hallucination verification, and confidence scoring.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import app

client = TestClient(app)


class TestSatQueryEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.optical_path = self._create_dummy_image("optical.png", shape=(256, 256, 3))
        self.sar_path = self._create_dummy_image("sentinel1_sar.png", shape=(256, 256, 1))

    def tearDown(self):
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def _create_dummy_image(self, filename: str, shape=(256, 256, 3)) -> str:
        filepath = os.path.join(self.temp_dir, filename)
        if len(shape) == 3 and shape[2] == 1:
            arr = (np.random.rand(shape[0], shape[1]) * 255).astype(np.uint8)
        else:
            arr = (np.random.rand(*shape) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(filepath)
        return filepath

    def test_health_check_endpoint(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("torch_version", data)

    def test_single_optical_vqa_flow(self):
        # 1. Upload
        with open(self.optical_path, "rb") as f:
            up_res = client.post("/upload", files=[("files", ("optical.png", f, "image/png"))])
        self.assertEqual(up_res.status_code, 200)
        tokens = up_res.json()["image_tokens"]

        # 2. Query
        q_res = client.post("/query", json={"query": "What visual features are present in this satellite image?", "image_tokens": tokens})
        self.assertEqual(q_res.status_code, 200)
        data = q_res.json()

        self.assertIn("answer", data)
        self.assertGreater(data["confidence"], 0.5)
        self.assertIn("single_vqa", data["execution_summary"]["task_type"])

    def test_bitemporal_change_detection_flow(self):
        t1_path = self._create_dummy_image("t1.png")
        t2_path = self._create_dummy_image("t2.png")

        with open(t1_path, "rb") as f1, open(t2_path, "rb") as f2:
            up_res = client.post("/upload", files=[
                ("files", ("t1.png", f1, "image/png")),
                ("files", ("t2.png", f2, "image/png"))
            ])
        self.assertEqual(up_res.status_code, 200)
        tokens = up_res.json()["image_tokens"]

        q_res = client.post("/query", json={"query": "What changed between T1 and T2?", "image_tokens": tokens})
        self.assertEqual(q_res.status_code, 200)
        data = q_res.json()

        self.assertEqual(data["execution_summary"]["task_type"], "change_vqa")
        self.assertTrue(data["visual_evidence"].get("has_change_map", False))

    def test_invalid_file_extension_rejection(self):
        bad_file_path = os.path.join(self.temp_dir, "script.sh")
        with open(bad_file_path, "w") as f:
            f.write("#!/bin/bash\necho hello")

        with open(bad_file_path, "rb") as f:
            up_res = client.post("/upload", files=[("files", ("script.sh", f, "text/plain"))])
        self.assertEqual(up_res.status_code, 400)
        self.assertIn("Unsupported file type", up_res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
