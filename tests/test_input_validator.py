"""
SatQuery AI — Test Suite for InputValidator Engine
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI
"""

import unittest
import os
import tempfile
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.input_validator import InputValidator, ValidationResult, ImageMetadata


class TestInputValidator(unittest.TestCase):

    def setUp(self):
        self.validator = InputValidator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp files
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def _create_dummy_png(self, filename: str, shape=(256, 256, 3)) -> str:
        filepath = os.path.join(self.temp_dir, filename)
        arr = (np.random.rand(*shape) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(filepath)
        return filepath

    def _create_dummy_sar_png(self, filename: str, shape=(256, 256)) -> str:
        filepath = os.path.join(self.temp_dir, filename)
        arr = (np.random.rand(*shape) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(filepath)
        return filepath

    def test_single_optical_validation(self):
        fp = self._create_dummy_png("optical_test.png")
        result = self.validator.validate_inputs([fp], query="What is in this image?")
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.num_images, 1)
        self.assertEqual(result.primary_modality, "single_optical")
        self.assertEqual(result.image_metadata[0].modality_type, "optical_rgb")
        self.assertIn("single_image_vqa", result.recommended_tools)

    def test_single_sar_validation(self):
        fp = self._create_dummy_sar_png("sentinel1_sar.png")
        result = self.validator.validate_inputs([fp], query="Identify structures in SAR image")
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.num_images, 1)
        self.assertEqual(result.primary_modality, "single_sar")
        self.assertEqual(result.image_metadata[0].modality_type, "sar")

    def test_bitemporal_change_validation(self):
        fp1 = self._create_dummy_png("t1_before.png")
        fp2 = self._create_dummy_png("t2_after.png")
        query = "What changed between T1 and T2?"
        
        result = self.validator.validate_inputs([fp1, fp2], query=query)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.num_images, 2)
        self.assertEqual(result.primary_modality, "bitemporal_optical")
        self.assertIn("change_detection", result.recommended_tools)
        self.assertIn("change_vqa", result.recommended_tools)

    def test_optical_sar_fusion_validation(self):
        fp_opt = self._create_dummy_png("optical_view.png")
        fp_sar = self._create_dummy_sar_png("sar_sentinel1.png")
        query = "Compare optical and radar view"
        
        result = self.validator.validate_inputs([fp_opt, fp_sar], query=query)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.num_images, 2)
        self.assertEqual(result.primary_modality, "optical_sar_pair")
        self.assertIn("optical_sar_fusion", result.recommended_tools)

    def test_sar_decibel_normalization(self):
        sar_amplitude = np.array([[0.1, 1.0], [10.0, 100.0]], dtype=np.float32)
        norm_sar = self.validator.normalize_sar(sar_amplitude, to_decibel=True)
        
        self.assertEqual(norm_sar.shape, (2, 2))
        self.assertTrue(np.all(norm_sar >= 0.0))
        self.assertTrue(np.all(norm_sar <= 1.0))

    def test_optical_band_normalization(self):
        optical_raw = np.array([[500, 2000], [5000, 10000]], dtype=np.float32)
        norm_opt = self.validator.normalize_optical(optical_raw, is_sentinel2_10k=True)
        
        self.assertEqual(norm_opt.shape, (2, 2))
        self.assertEqual(norm_opt[0, 0], 0.05)
        self.assertEqual(norm_opt[1, 1], 1.0)

    def test_corrupt_file_handling(self):
        corrupt_path = os.path.join(self.temp_dir, "corrupt.png")
        with open(corrupt_path, "w") as f:
            f.write("Not an image file!")
            
        result = self.validator.validate_inputs([corrupt_path])
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.validation_errors), 0)


if __name__ == "__main__":
    unittest.main()
