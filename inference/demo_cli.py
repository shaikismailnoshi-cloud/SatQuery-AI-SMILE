"""
SatQuery AI — Interactive Command-Line Evaluator Demo Tool
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Command-line demonstration script allowing evaluators to run queries against satellite imagery,
inspect GeoTIFF metadata, view tool routing, and observe calibrated confidence scores.
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.input_validator import InputValidator
from agents.agentic_router import AgenticRouter
from agents.evidence_engine import EvidenceFusionEngine
from inference.inference_engine import SatQueryInferenceEngine


def run_cli_demo(image_paths: list, query: str):
    print("\n" + "="*70)
    print("  🛰️ SATQUERY AI — INTERACTIVE EVALUATOR CLI DEMO (SIH26167)")
    print("="*70)

    # 1. Validate Inputs & Metadata
    validator = InputValidator()
    val_res = validator.validate_inputs(image_paths, query=query)
    
    if not val_res.is_valid:
        print("\n❌ INPUT VALIDATION ERROR:")
        for err in val_res.validation_errors:
            print(f"  - {err}")
        return

    print("\n1. GEOSPATIAL & SENSOR METADATA INSPECTION:")
    print(f"  • Number of Images: {val_res.num_images}")
    print(f"  • Primary Modality: {val_res.primary_modality.upper()}")
    for idx, meta in enumerate(val_res.image_metadata, start=1):
        print(f"  • Image {idx}: {meta.filename}")
        print(f"    - Format: {meta.file_format} | Bands: {meta.num_bands} | Dimensions: {meta.width}x{meta.height}")
        print(f"    - CRS: {meta.crs if meta.crs else 'Standard RGB'} | Bounds: {meta.bounds if meta.bounds else 'None'}")
        print(f"    - Modality Inferred: {meta.modality_type}")

    if val_res.compatibility_warnings:
        print("\n⚠️ COMPATIBILITY WARNINGS:")
        for warn in val_res.compatibility_warnings:
            print(f"  - {warn}")

    # 2. Agentic Routing
    router = AgenticRouter()
    plan = router.route_request(query, val_res.to_dict())

    print("\n2. AGENTIC ROUTER & TOOL ORCHESTRATION:")
    print(f"  • User Query: '{query}'")
    print(f"  • Identified Task Type: {plan.task_type.upper()}")
    print(f"  • Routing Decision: {plan.reasoning_summary}")
    print("  • Selected Specialist Tools:")
    for step in plan.execution_order:
        print(f"    - Step {step['step']}: {step['tool_name']} ({step['tool']})")

    # 3. PyTorch Tensor Inference Engine Execution
    print("\n3. PYTORCH TENSOR INFERENCE & FEATURE EXTRACTION:")
    engine = SatQueryInferenceEngine()
    
    primary_img = image_paths[0]
    secondary_img = image_paths[1] if len(image_paths) > 1 else primary_img

    specialist_outputs = {}
    if "single_image_vqa" in plan.selected_tools:
        specialist_outputs["single_image_vqa"] = engine.run_single_vqa(primary_img, query)
    if "captioning" in plan.selected_tools:
        specialist_outputs["captioning"] = engine.run_captioning(primary_img)
    if "grounding" in plan.selected_tools:
        specialist_outputs["grounding"] = engine.run_grounding(primary_img, query)
    if "change_detection" in plan.selected_tools or "change_vqa" in plan.selected_tools:
        specialist_outputs["change_vqa"] = engine.run_change_detection(primary_img, secondary_img)
    if "optical_sar_fusion" in plan.selected_tools:
        specialist_outputs["optical_sar_fusion"] = engine.run_optical_sar_fusion(primary_img, secondary_img, query)

    # 4. Evidence Fusion & Anti-Hallucination
    evidence_engine = EvidenceFusionEngine()
    bundle = evidence_engine.fuse_and_verify(query, specialist_outputs, plan.selected_tools)

    print("\n4. FINAL RESPONSE CARD:")
    print(f"  • Answer: {bundle.raw_answer}")
    print(f"  • Calibrated Confidence: {bundle.calibrated_confidence:.1%} ({bundle.confidence_level})")
    print(f"  • Uncertainty Score: {bundle.uncertainty_score:.3f}")
    if bundle.hallucination_warning:
        print(f"  • Anti-Hallucination Alert: ⚠️ {bundle.hallucination_warning}")
    print("  • Supporting Visual & Spectral Evidence:")
    for ev in bundle.evidence_items:
        print(f"    - {ev}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI CLI Evaluator Tool")
    parser.add_argument("--images", nargs="+", required=True, help="List of satellite image filepaths (GeoTIFF, TIFF, PNG)")
    parser.add_argument("--query", type=str, default="Describe this satellite image scene", help="Natural language query")
    args = parser.parse_args()
    
    run_cli_demo(args.images, args.query)
