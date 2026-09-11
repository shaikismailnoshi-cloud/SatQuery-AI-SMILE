"""
SatQuery AI — Task-to-Evidence Validator & Source Requirement Recommendation Engine
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Analyses uploaded imagery properties against requested ML tasks before model execution.
Enforces evidence-driven reasoning, prevents language-prior hallucination, and generates
actionable source recommendations when imagery is insufficient or unsuitable.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("satquery.source_requirement")


@dataclass
class SourceRequirement:
    status: str  # "VALID", "PARTIALLY_VALID", "INSUFFICIENT", "INVALID"
    requested_task: str
    image_analysis: str  # Human-readable summary of what is actually visible
    missing_requirement: str
    recommended_source: str
    why: str
    next_action: str
    suitability_score: float
    evidence_available: bool
    visible_content: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SourceRequirementEngine:
    """
    Source Recommendation Engine & Task-to-Evidence Validator.
    Maps requested ML tasks to required visual/spectral evidence and produces
    structured guidance when evidence is lacking.
    """

    TASK_REQUIREMENTS = {
        "building_detection": {
            "name": "Building Structure Detection & Counting",
            "required_evidence": "Visible rectilinear building structures, rooftops, or impervious surface reflectance.",
            "recommended_source": "High-resolution optical satellite imagery (spatial resolution < 1.0m/pixel) with clear rooftop visibility.",
            "why": "Building detection requires fine spatial resolution to resolve individual structural boundaries. Building detection cannot be reliably performed on non-urban, low-resolution, or featureless scenes.",
            "next_action": "Please upload a high-resolution optical satellite image covering urban, residential, or industrial areas."
        },
        "water_detection": {
            "name": "Water Body Identification & Delineation",
            "required_evidence": "Visible spectral contrast in blue/green channels or low SAR microwave backscatter absorption.",
            "recommended_source": "Optical, multispectral, or SAR satellite imagery covering surface water features.",
            "why": "Water body detection requires distinct spectral absorption or smooth radar surface reflection.",
            "next_action": "Please upload an optical or SAR satellite image covering lakes, rivers, or coastal regions."
        },
        "vegetation_analysis": {
            "name": "Vegetation & Agricultural Health Analysis",
            "required_evidence": "Visible greenness reflectance (ExG) or Near-Infrared (NIR) band data.",
            "recommended_source": "Multispectral imagery (Red and NIR bands) or clear optical RGB satellite imagery.",
            "why": "Vegetation analysis requires measurable chlorophyll reflectance (ExG / NDVI spectral index).",
            "next_action": "Please upload a multispectral or high-quality optical satellite image of vegetated terrain."
        },
        "sar_interpretation": {
            "name": "SAR Radar Backscatter & Structural Analysis",
            "required_evidence": "Microwave backscatter intensity (VV/VH polarizations) or single-channel radar raster.",
            "recommended_source": "Synthetic Aperture Radar (SAR) imagery (Sentinel-1, RISAT-1, ICEYE, or RADARSAT).",
            "why": "SAR analysis relies on active microwave backscatter signatures and double-bounce reflections, which are not present in standard optical RGB imagery.",
            "next_action": "Please upload a Synthetic Aperture Radar (SAR) image file."
        },
        "change_detection": {
            "name": "Bi-Temporal Change Analysis",
            "required_evidence": "Two co-registered multi-temporal satellite images (T1 before and T2 after) of the same area.",
            "recommended_source": "A temporal pair of compatible satellite images acquired over the same geographic location at different dates.",
            "why": "Change detection requires comparing two temporal observations of the exact same spatial footprint to measure pixel-level alterations.",
            "next_action": "Please upload a second co-registered satellite image from a different acquisition date."
        },
        "road_detection": {
            "name": "Road Infrastructure & Transport Corridor Extraction",
            "required_evidence": "Continuous linear impervious pavement features with high spatial continuity.",
            "recommended_source": "High-resolution optical satellite imagery (spatial resolution < 1.5m/pixel).",
            "why": "Road network extraction requires clear spatial continuity of linear transportation corridors.",
            "next_action": "Please upload a high-resolution optical satellite image containing visible transportation networks."
        },
        "general_land_cover": {
            "name": "Land-Cover Classification & Scene Description",
            "required_evidence": "Discernible land surface features (vegetation, water, soil, or built structures).",
            "recommended_source": "Optical or multispectral satellite imagery.",
            "why": "General scene description requires clear surface reflectance features.",
            "next_action": "Please upload a clear satellite image of the target area."
        }
    }

    def evaluate_task_suitability(
        self,
        query: str,
        validation_result: Dict[str, Any],
        content_features: Dict[str, Any]
    ) -> SourceRequirement:
        """
        Evaluates whether the uploaded imagery contains sufficient visual/spectral evidence
        to satisfy the requested ML task.
        """
        q_lower = query.lower().strip()
        num_imgs = validation_result.get("num_images", 1)
        modality = validation_result.get("primary_modality", "single_optical")
        
        # 1. Identify Requested Task Type
        task_type = self._identify_task_type(q_lower, num_imgs, modality)
        req_info = self.TASK_REQUIREMENTS.get(task_type, self.TASK_REQUIREMENTS["general_land_cover"])
        
        # 2. Extract Visible Content Breakdown & Quality Metrics
        visible_summary_parts = []
        is_sar = content_features.get("is_sar", False)
        water_pct = content_features.get("water_pct", 0.0)
        veg_pct = content_features.get("veg_pct", 0.0)
        grey_pct = content_features.get("grey_impervious_pct", 0.0)
        soil_pct = content_features.get("soil_pct", 0.0)
        cloud_pct = content_features.get("cloud_pct", 0.0)
        blur_score = content_features.get("blur_score", 1.0)
        
        building_detected = content_features.get("building_detected", False)
        building_count = content_features.get("building_count", 0)
        
        if is_sar:
            high_bs = content_features.get("high_backscatter_pct", 0.0)
            low_bs = content_features.get("low_backscatter_pct", 0.0)
            visible_summary_parts.append(f"SAR Radar Imagery (High Specular Backscatter: {high_bs:.1f}%, Low Absorption: {low_bs:.1f}%)")
        else:
            if water_pct > 5.0:
                visible_summary_parts.append(f"Water Body (~{water_pct:.1f}%)")
            if veg_pct > 10.0:
                visible_summary_parts.append(f"Vegetation (~{veg_pct:.1f}%)")
            if soil_pct > 15.0:
                visible_summary_parts.append(f"Bare Soil/Rock (~{soil_pct:.1f}%)")
            if grey_pct > 3.0 or building_detected:
                visible_summary_parts.append(f"Impervious/Urban Structures (~{grey_pct:.1f}%, Detected: {building_count})")
            if cloud_pct > 15.0:
                visible_summary_parts.append(f"Cloud Cover (~{cloud_pct:.1f}%)")
                
        if not visible_summary_parts:
            visible_summary_parts.append("Uniform/low-contrast surface reflectance")

        image_analysis_text = f"The uploaded image contains: {', '.join(visible_summary_parts)}."

        visible_content = {
            "is_sar": is_sar,
            "water_pct": round(water_pct, 1),
            "veg_pct": round(veg_pct, 1),
            "soil_pct": round(soil_pct, 1),
            "grey_impervious_pct": round(grey_pct, 1),
            "cloud_pct": round(cloud_pct, 1),
            "building_detected": building_detected,
            "building_count": building_count,
            "quality_acceptable": cloud_pct < 40.0 and blur_score > 0.15
        }

        # 3. Task-to-Evidence Validation Rules
        status = "VALID"
        missing_requirement = "None. The image contains sufficient visual evidence."
        why = f"The image features align with the requirements for {req_info['name']}."
        next_action = "Proceeding with ML model execution."
        suitability_score = 0.90
        evidence_available = True

        # Rule A: Heavy Cloud Obstruction (> 45% clouds)
        if cloud_pct > 45.0:
            status = "INSUFFICIENT"
            evidence_available = False
            suitability_score = 0.15
            missing_requirement = f"Heavy cloud cover (~{cloud_pct:.1f}%) obstructs surface features."
            why = "Atmospheric cloud obstruction prevents reliable optical analysis of ground objects."
            next_action = "Please upload a cloud-free optical image or a cloud-penetrating SAR image."

        # Rule B: Building Detection Task vs. Water-Only or Empty Image
        elif task_type == "building_detection":
            if not building_detected and grey_pct < 2.5:
                status = "INSUFFICIENT"
                evidence_available = False
                suitability_score = 0.10
                missing_requirement = "No clearly detectable building rooftops or structural contours visible in imagery."
                if water_pct > 50.0:
                    why = f"The uploaded image appears to be dominated by a water body (~{water_pct:.1f}%), but no clearly detectable buildings are visible. Building detection cannot be reliably performed on this image."
                elif veg_pct > 70.0:
                    why = f"The uploaded image is dominated by dense natural vegetation (~{veg_pct:.1f}%), without discernible building structures."
                else:
                    why = "No structural building footprints or rectilinear rooftop contours were detected in this image."
                next_action = req_info["next_action"]
            elif building_count > 0 and building_count < 3:
                status = "PARTIALLY_VALID"
                suitability_score = 0.65
                missing_requirement = "Low density of building structures; sparse detection."

        # Rule C: Bi-Temporal Change Query with Single Image
        elif task_type == "change_detection" and num_imgs < 2:
            status = "INSUFFICIENT"
            evidence_available = False
            suitability_score = 0.05
            missing_requirement = "Second temporal image (T2) missing."
            why = "Change detection requires a pair of co-registered satellite images acquired at different times."
            next_action = req_info["next_action"]

        # Rule D: SAR Analysis Requested on Pure Optical Image
        elif task_type == "sar_interpretation" and not is_sar:
            status = "INSUFFICIENT"
            evidence_available = False
            suitability_score = 0.10
            missing_requirement = "SAR microwave polarization data missing."
            why = "SAR analysis requires active radar backscatter rasters, but an optical RGB image was provided."
            next_action = req_info["next_action"]

        # Rule E: Water Detection on Image with 0% Water Evidence
        elif task_type == "water_detection" and water_pct < 0.5 and not is_sar:
            status = "INSUFFICIENT"
            evidence_available = False
            suitability_score = 0.20
            missing_requirement = "No surface water spectral reflection identified."
            why = "No water bodies or blue-dominant absorption features are visible in the uploaded image."
            next_action = req_info["next_action"]

        # Rule F: Vegetation Analysis on 0% Vegetation Scene
        elif task_type == "vegetation_analysis" and veg_pct < 1.0 and not content_features.get("ndvi_available", False):
            status = "INSUFFICIENT"
            evidence_available = False
            suitability_score = 0.20
            missing_requirement = "No vegetation spectral signal (ExG / NDVI) detected."
            why = "Natural vegetation signatures are absent in the provided imagery."
            next_action = req_info["next_action"]

        return SourceRequirement(
            status=status,
            requested_task=query,
            image_analysis=image_analysis_text,
            missing_requirement=missing_requirement,
            recommended_source=req_info["recommended_source"],
            why=why,
            next_action=next_action,
            suitability_score=round(suitability_score, 2),
            evidence_available=evidence_available,
            visible_content=visible_content
        )

    def _identify_task_type(self, q_lower: str, num_imgs: int, modality: str) -> str:
        """Determines requested ML task category from query semantics."""
        if any(k in q_lower for k in ["building", "urban", "structure", "city", "house", "settlement", "count"]):
            return "building_detection"
        if any(k in q_lower for k in ["water", "river", "lake", "ocean", "pond", "flood"]):
            return "water_detection"
        if any(k in q_lower for k in ["vegetation", "forest", "tree", "crop", "agriculture", "green", "ndvi"]):
            return "vegetation_analysis"
        if any(k in q_lower for k in ["sar", "radar", "backscatter", "polarization", "microwave"]):
            return "sar_interpretation"
        if num_imgs >= 2 or any(k in q_lower for k in ["change", "differ", "before", "after", "increased", "decreased"]):
            return "change_detection"
        if any(k in q_lower for k in ["road", "highway", "street", "transport"]):
            return "road_detection"
        return "general_land_cover"

