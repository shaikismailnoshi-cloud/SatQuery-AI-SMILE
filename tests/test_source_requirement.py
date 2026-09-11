"""
SatQuery AI — Unit Tests for Source Requirement & Task-to-Evidence Validator
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI
"""

import unittest
from agents.source_requirement import SourceRequirementEngine, SourceRequirement
from agents.evidence_engine import EvidenceFusionEngine


class TestSourceRequirementEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SourceRequirementEngine()
        self.evidence_engine = EvidenceFusionEngine()

    def test_water_image_building_query_rejection(self):
        """Uploading water image and asking for building count must report INSUFFICIENT evidence, not hallucinate."""
        query = "Find and count all buildings in this scene."
        val_dict = {"num_images": 1, "primary_modality": "single_optical"}
        content_feats = {
            "is_sar": False,
            "water_pct": 85.5,
            "veg_pct": 5.0,
            "grey_impervious_pct": 0.2,
            "soil_pct": 2.0,
            "cloud_pct": 0.0,
            "building_detected": False,
            "building_count": 0
        }

        res = self.engine.evaluate_task_suitability(query, val_dict, content_feats)
        self.assertEqual(res.status, "INSUFFICIENT")
        self.assertFalse(res.evidence_available)
        self.assertIn("water body", res.why.lower())
        self.assertIn("optical satellite imagery", res.recommended_source.lower())

        # Test EvidenceFusionEngine output
        bundle = self.evidence_engine.fuse_and_verify(query, {}, ["single_image_vqa"], source_req=res)
        self.assertFalse(bundle.is_evidence_sufficient)
        self.assertEqual(bundle.evidence_status, "INSUFFICIENT")
        self.assertIn("Insufficient visual evidence", bundle.raw_answer)

    def test_valid_building_detection(self):
        """Uploading image with building footprints returns VALID status."""
        query = "Locate all buildings."
        val_dict = {"num_images": 1, "primary_modality": "single_optical"}
        content_feats = {
            "is_sar": False,
            "water_pct": 2.0,
            "veg_pct": 10.0,
            "grey_impervious_pct": 25.0,
            "soil_pct": 5.0,
            "cloud_pct": 0.0,
            "building_detected": True,
            "building_count": 8
        }

        res = self.engine.evaluate_task_suitability(query, val_dict, content_feats)
        self.assertEqual(res.status, "VALID")
        self.assertTrue(res.evidence_available)

    def test_change_detection_single_image_rejection(self):
        """Asking for change detection with only 1 image returns INSUFFICIENT with temporal pair requirement."""
        query = "What changes occurred between these two satellite images?"
        val_dict = {"num_images": 1, "primary_modality": "single_optical"}
        content_feats = {"is_sar": False, "water_pct": 10.0, "veg_pct": 20.0, "grey_impervious_pct": 5.0, "cloud_pct": 0.0}

        res = self.engine.evaluate_task_suitability(query, val_dict, content_feats)
        self.assertEqual(res.status, "INSUFFICIENT")
        self.assertIn("second", res.next_action.lower())

    def test_sar_query_on_optical_image_rejection(self):
        """Asking for SAR radar analysis on optical imagery requests SAR data source."""
        query = "Analyze SAR microwave backscatter and polarization."
        val_dict = {"num_images": 1, "primary_modality": "single_optical"}
        content_feats = {"is_sar": False, "water_pct": 10.0, "veg_pct": 20.0, "grey_impervious_pct": 5.0, "cloud_pct": 0.0}

        res = self.engine.evaluate_task_suitability(query, val_dict, content_feats)
        self.assertEqual(res.status, "INSUFFICIENT")
        self.assertIn("sar", res.recommended_source.lower())

    def test_cloud_obstruction_rejection(self):
        """Images with > 45% cloud cover report cloud obstruction rejection."""
        query = "Identify land cover and structures."
        val_dict = {"num_images": 1, "primary_modality": "single_optical"}
        content_feats = {"is_sar": False, "water_pct": 0.0, "veg_pct": 10.0, "grey_impervious_pct": 5.0, "cloud_pct": 62.0}

        res = self.engine.evaluate_task_suitability(query, val_dict, content_feats)
        self.assertEqual(res.status, "INSUFFICIENT")
        self.assertIn("cloud", res.missing_requirement.lower())


if __name__ == "__main__":
    unittest.main()
