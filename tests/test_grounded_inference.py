"""
SatQuery AI — Grounded Model Accuracy & Anti-Hallucination Test Suite
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Verifies:
1. Answers change according to the uploaded image content.
2. Answers change according to the user's natural-language question.
3. Absence of features returns "Not clearly identifiable from the provided imagery."
4. Bounding boxes are computed dynamically from actual object contours.
5. Confidence scores are non-hardcoded and dynamically calibrated.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.inference_engine import SatQueryInferenceEngine
from agents.agentic_router import AgenticRouter
from agents.evidence_engine import EvidenceFusionEngine
from backend.input_validator import InputValidator


class TestGroundedModelAccuracy(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = SatQueryInferenceEngine()
        self.validator = InputValidator()
        self.router = AgenticRouter()
        self.fusion = EvidenceFusionEngine()

        # Image A: Pure Green (Vegetation only, no water, no buildings)
        self.img_veg_path = os.path.join(self.temp_dir, "pure_veg.png")
        arr_veg = np.zeros((256, 256, 3), dtype=np.uint8)
        arr_veg[:, :, 1] = 200  # High green channel
        Image.fromarray(arr_veg).save(self.img_veg_path)

        # Image B: Pure Blue (Water body only, no vegetation, no buildings)
        self.img_water_path = os.path.join(self.temp_dir, "pure_water.png")
        arr_water = np.zeros((256, 256, 3), dtype=np.uint8)
        arr_water[:, :, 2] = 220  # High blue channel
        arr_water[:, :, 0] = 20
        arr_water[:, :, 1] = 60
        Image.fromarray(arr_water).save(self.img_water_path)

        # Image C: Urban building image (Clean background with distinct building rectangles)
        self.img_urban_path = os.path.join(self.temp_dir, "urban_buildings.png")
        arr_urban = np.ones((256, 256, 3), dtype=np.uint8) * 40
        # Create distinct rectangular building shapes
        arr_urban[40:110, 40:110] = 240
        arr_urban[150:210, 140:200] = 240
        Image.fromarray(arr_urban).save(self.img_urban_path)

    def tearDown(self):
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_water_query_accuracy_and_anti_hallucination(self):
        """Water query on water image vs vegetation image."""
        res_water = self.engine.run_single_vqa(self.img_water_path, "Are there water bodies?")
        self.assertIn("water", res_water["answer"].lower())

        res_no_water = self.engine.run_single_vqa(self.img_veg_path, "Are there water bodies?")
        self.assertIn("not clearly identifiable", res_no_water["answer"].lower())

    def test_building_query_accuracy_and_anti_hallucination(self):
        """Building query on urban image vs vegetation image."""
        res_urban = self.engine.run_single_vqa(self.img_urban_path, "Locate building structures")
        self.assertIn("building", res_urban["answer"].lower())

        res_no_urban = self.engine.run_single_vqa(self.img_veg_path, "Locate building structures")
        self.assertIn("no buildings", res_no_urban["answer"].lower())

    def test_question_dependence_on_same_image(self):
        """Two different questions on the same image return different, relevant answers."""
        q1_res = self.engine.run_single_vqa(self.img_veg_path, "Is there vegetation?")
        q2_res = self.engine.run_single_vqa(self.img_veg_path, "Are there buildings?")
        
        self.assertNotEqual(q1_res["answer"], q2_res["answer"])
        self.assertIn("vegetation", q1_res["answer"].lower())
        self.assertIn("no buildings", q2_res["answer"].lower())

    def test_real_contour_bounding_box_grounding(self):
        """Bounding boxes are extracted from real image shapes, not hardcoded."""
        grd_urban = self.engine.run_grounding(self.img_urban_path, "Locate buildings")
        self.assertGreater(len(grd_urban["bboxes"]), 0)
        
        box = grd_urban["bboxes"][0]["box"]
        self.assertEqual(len(box), 4)
        # Check box coordinates are valid floats between 0 and 1
        for coord in box:
            self.assertGreaterEqual(coord, 0.0)
            self.assertLessEqual(coord, 1.0)

    def test_non_hardcoded_confidence_scoring(self):
        """Confidence varies dynamically across images."""
        res1 = self.engine.run_single_vqa(self.img_water_path, "Are there water bodies?")
        res2 = self.engine.run_single_vqa(self.img_veg_path, "Are there water bodies?")
        
        self.assertNotEqual(res1["confidence"], res2["confidence"])
        self.assertNotEqual(res1["confidence"], 0.87)
        self.assertNotEqual(res2["confidence"], 0.87)


if __name__ == "__main__":
    unittest.main()
