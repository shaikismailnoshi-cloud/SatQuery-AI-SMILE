"""
SatQuery AI — Agentic Router & Tool Orchestration Engine
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Parses user query semantics and image metadata (modality, count, CRS, temporal relationship)
to dynamically select specialist tools, determine execution workflow, and build auditable trace records.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("satquery.agentic_router")


@dataclass
class ToolExecutionPlan:
    task_type: str  # "single_vqa", "captioning", "grounding", "change_detection", "change_vqa", "optical_sar_fusion"
    selected_tools: List[str]
    primary_modality: str
    num_images: int
    execution_order: List[Dict[str, Any]]
    reasoning_summary: str  # High-level clean summary for trace audit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgenticRouter:
    """
    Agentic Router for SatQuery AI.
    Routes queries to specialist models based on query intent and imagery properties.
    """

    KNOWN_SPECIALISTS = {
        "input_validator": "Geospatial Image Ingestion & Metadata Validator",
        "source_requirement_engine": "Task-to-Evidence Validator & Source Requirement Engine",
        "single_image_vqa": "RS Single-Image Visual Question Answering Model",
        "captioning": "Remote Sensing Spatial-Spectral Scene Description Model",
        "grounding": "Visual Bounding Box Grounding & Localization Engine",
        "change_detection": "Bi-Temporal Pixel Change Detection & Mask Generator",
        "change_vqa": "Temporal Reasoning & Change VQA Specialist",
        "optical_sar_fusion": "Dual-Stream Optical-SAR Gated Cross-Attention Encoder",
        "evidence_engine": "Structured Multi-Model Evidence & Anti-Hallucination Fusion",
        "confidence_engine": "Calibrated Temperature-Scaled Uncertainty Estimator"
    }

    def route_request(self, query: str, validation_result: Dict[str, Any]) -> ToolExecutionPlan:
        """
        Main routing entrypoint. Accepts natural-language query and metadata validation output.
        """
        q_lower = query.lower().strip()
        num_imgs = validation_result.get("num_images", 1)
        modality = validation_result.get("primary_modality", "single_optical")
        
        selected_tools = []
        task_type = "single_vqa"

        # Rule 1: Change / Temporal Queries or Multi-temporal images
        is_change_query = any(k in q_lower for k in ["change", "differ", "before", "after", "increased", "decreased", "new", "removed", "built", "lost"])
        if num_imgs >= 2 or is_change_query:
            if is_change_query or modality in {"bitemporal_optical", "bitemporal_sar"}:
                task_type = "change_vqa"
                selected_tools = ["change_detection", "change_vqa", "evidence_engine", "confidence_engine"]
                summary = "Detected multi-temporal query/inputs. Routing to Bi-Temporal Change Detection and Change VQA specialists."
            elif modality == "optical_sar_pair" or "sar" in q_lower or "radar" in q_lower:
                task_type = "optical_sar_fusion"
                selected_tools = ["optical_sar_fusion", "single_image_vqa", "evidence_engine", "confidence_engine"]
                summary = "Detected co-registered Optical + SAR dual modality. Routing to Gated Optical-SAR Fusion encoder."
            else:
                task_type = "change_vqa"
                selected_tools = ["change_detection", "single_image_vqa", "evidence_engine", "confidence_engine"]
                summary = "Detected multi-image input. Routing to Change Detection and Comparative VQA engine."

        # Rule 2: Grounding / Bounding Box Queries
        elif any(k in q_lower for k in ["where", "locate", "find", "bounding box", "bbox", "coordinates", "show me"]):
            task_type = "grounding"
            selected_tools = ["grounding", "single_image_vqa", "evidence_engine", "confidence_engine"]
            summary = "Detected spatial localization query. Routing to Visual Grounding Engine and BBox Regressor."

        # Rule 3: Captioning / Overview Queries
        elif any(k in q_lower for k in ["describe", "caption", "overview", "summary", "detail", "tell me about"]):
            task_type = "captioning"
            selected_tools = ["captioning", "evidence_engine", "confidence_engine"]
            summary = "Detected scene description query. Routing to Remote Sensing Captioner."

        # Rule 4: Optical-SAR specific single/dual queries
        elif modality in {"single_sar", "optical_sar_pair"} or "radar" in q_lower:
            task_type = "optical_sar_fusion"
            selected_tools = ["optical_sar_fusion", "single_image_vqa", "evidence_engine", "confidence_engine"]
            summary = "Detected SAR imagery input. Routing to SAR-specialized encoder and multimodal VQA engine."

        # Rule 5: Default Single Image VQA
        else:
            task_type = "single_vqa"
            selected_tools = ["single_image_vqa", "evidence_engine", "confidence_engine"]
            summary = "Routing to Single-Image Remote-Sensing VQA Specialist."

        # Construct Execution Order
        execution_order = []
        for idx, tool in enumerate(selected_tools, start=1):
            execution_order.append({
                "step": idx,
                "tool": tool,
                "tool_name": self.KNOWN_SPECIALISTS.get(tool, tool),
                "status": "planned"
            })

        return ToolExecutionPlan(
            task_type=task_type,
            selected_tools=selected_tools,
            primary_modality=modality,
            num_images=num_imgs,
            execution_order=execution_order,
            reasoning_summary=summary
        )
