"""
SatQuery AI — Structured Evidence Engine & Calibrated Confidence Estimator
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Aggregates prediction evidence from visual specialists, checks model consensus,
prevents language-prior hallucination, and computes temperature-scaled confidence scores.
"""

import math
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("satquery.evidence_engine")


@dataclass
class EvidenceBundle:
    raw_answer: str
    calibrated_confidence: float
    confidence_level: str  # "HIGH", "MEDIUM", "LOW"
    uncertainty_score: float
    evidence_items: List[str]
    visual_evidence: Dict[str, Any]  # BBox, Change map summary, etc.
    models_used: List[str]
    hallucination_warning: Optional[str]
    is_evidence_sufficient: bool
    evidence_status: str = "VALID"  # "VALID", "PARTIALLY_VALID", "INSUFFICIENT", "INVALID"
    image_suitability_score: float = 0.90
    source_requirement: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceFusionEngine:
    """
    Evidence Fusion & Verification Engine.
    Enforces grounding in imagery data and prevents LLM hallucination.
    """

    def fuse_and_verify(
        self,
        query: str,
        specialist_outputs: Dict[str, Any],
        selected_tools: List[str],
        source_req: Optional[Any] = None
    ) -> EvidenceBundle:
        """
        Consolidates outputs from specialist models, checks evidence consensus,
        and computes calibrated confidence.
        """
        evidence_items = []
        visual_evidence = {}
        models_used = selected_tools
        
        raw_answer = ""
        conf_scores = []
        conflicting_evidence = False
        insufficient_evidence = False
        hallucination_warning = None
        source_req_dict = source_req.to_dict() if source_req and hasattr(source_req, "to_dict") else None

        # 0. Pre-check Source Requirement Validation
        if source_req and getattr(source_req, "status", "VALID") in {"INSUFFICIENT", "INVALID"}:
            insufficient_evidence = True
            raw_answer = f"Insufficient visual evidence. {source_req.why}"
            hallucination_warning = f"Required source missing: {source_req.recommended_source}"
            evidence_items.append(f"Source Requirement Engine: Status {source_req.status}. {source_req.missing_requirement}")
            evidence_items.append(f"Image Analysis: {source_req.image_analysis}")

            return EvidenceBundle(
                raw_answer=raw_answer,
                calibrated_confidence=0.15,
                confidence_level="LOW",
                uncertainty_score=0.85,
                evidence_items=evidence_items,
                visual_evidence={"bounding_boxes": [], "has_change_map": False},
                models_used=models_used,
                hallucination_warning=hallucination_warning,
                is_evidence_sufficient=False,
                evidence_status=source_req.status,
                image_suitability_score=source_req.suitability_score,
                source_requirement=source_req_dict
            )

        # 1. Inspect Single Image VQA Output
        if "single_image_vqa" in specialist_outputs:
            vqa_out = specialist_outputs["single_image_vqa"]
            ans = vqa_out.get("answer", "")
            sc = vqa_out.get("confidence", 0.8)
            conf_scores.append(sc)
            raw_answer = ans
            evidence_items.append(f"VQA Model Prediction: '{ans}' (Raw Confidence: {sc:.2f})")

        # 2. Inspect Captioner Output
        if "captioning" in specialist_outputs:
            cap_out = specialist_outputs["captioning"]
            caption = cap_out.get("caption", "")
            sc = cap_out.get("confidence", 0.85)
            conf_scores.append(sc)
            if not raw_answer:
                raw_answer = caption
            evidence_items.append(f"Scene Description: '{caption}'")

        # 3. Inspect Grounding Output
        if "grounding" in specialist_outputs:
            grd_out = specialist_outputs["grounding"]
            bboxes = grd_out.get("bboxes", [])
            visual_evidence["bounding_boxes"] = bboxes
            if len(bboxes) > 0:
                evidence_items.append(f"Detected Bounding Boxes: {len(bboxes)} spatial targets located.")
                if not raw_answer:
                    raw_answer = f"Located {len(bboxes)} spatial target(s) matching query."
            else:
                evidence_items.append("Visual Grounding Analysis: No matching target contours detected.")
                if not raw_answer:
                    raw_answer = "Target features matching the query are not clearly identifiable from the provided imagery."

        # 4. Inspect Change Detection & Change VQA Output
        if "change_detection" in specialist_outputs or "change_vqa" in specialist_outputs:
            chg_out = specialist_outputs.get("change_vqa", specialist_outputs.get("change_detection", {}))
            chg_ans = chg_out.get("change_description", chg_out.get("answer", ""))
            chg_pct = chg_out.get("changed_area_pct", 0.0)
            sc = chg_out.get("confidence", 0.82)
            conf_scores.append(sc)
            visual_evidence["changed_area_pct"] = chg_pct
            visual_evidence["has_change_map"] = True
            if chg_ans:
                raw_answer = chg_ans
            evidence_items.append(f"Bi-Temporal Change Analysis: {chg_pct:.1f}% area changed. Answer: '{chg_ans}'")

        # 5. Inspect Optical-SAR Fusion Output
        if "optical_sar_fusion" in specialist_outputs:
            fus_out = specialist_outputs["optical_sar_fusion"]
            fus_ans = fus_out.get("fused_answer", "")
            sc = fus_out.get("confidence", 0.88)
            conf_scores.append(sc)
            evidence_items.append(f"Optical-SAR Dual Stream Fusion: '{fus_ans}'")
            if fus_ans:
                raw_answer = fus_ans

        # 6. Anti-Hallucination & Evidence Consensus Verification
        if not conf_scores or not raw_answer:
            insufficient_evidence = True
            raw_answer = "Insufficient visual evidence to determine this reliably."
            hallucination_warning = "No specialist model produced high-confidence visual features for this prompt."
            calibrated_conf = 0.20
        else:
            avg_score = float(sum(conf_scores) / len(conf_scores))
            # Check for conflict between multiple tools
            if len(conf_scores) > 1 and (max(conf_scores) - min(conf_scores)) > 0.35:
                conflicting_evidence = True
                hallucination_warning = "The available models provide conflicting evidence, so confidence is reduced."
                calibrated_conf = max(0.30, avg_score * 0.70)
            else:
                calibrated_conf = self._temperature_scale(avg_score, temperature=1.2)

        # 7. Map Confidence & Uncertainty Badges
        if calibrated_conf >= 0.80:
            level = "HIGH"
        elif calibrated_conf >= 0.50:
            level = "MEDIUM"
        else:
            level = "LOW"

        uncertainty_score = float(np_clip(1.0 - calibrated_conf, 0.0, 1.0))
        ev_status = source_req.status if source_req else ("PARTIALLY_VALID" if hallucination_warning else "VALID")
        suitability = source_req.suitability_score if source_req else 0.90

        return EvidenceBundle(
            raw_answer=raw_answer,
            calibrated_confidence=round(calibrated_conf, 3),
            confidence_level=level,
            uncertainty_score=round(uncertainty_score, 3),
            evidence_items=evidence_items,
            visual_evidence=visual_evidence,
            models_used=models_used,
            hallucination_warning=hallucination_warning,
            is_evidence_sufficient=not insufficient_evidence,
            evidence_status=ev_status,
            image_suitability_score=suitability,
            source_requirement=source_req_dict
        )

    def _temperature_scale(self, prob: float, temperature: float = 1.2) -> float:
        """Applies temperature scaling to raw model softmax probabilities."""
        logit = math.log(max(prob, 1e-6) / max(1.0 - prob, 1e-6))
        scaled_logit = logit / temperature
        return 1.0 / (1.0 + math.exp(-scaled_logit))


def np_clip(val: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, val))

