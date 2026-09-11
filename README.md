# SatQuery AI 🛰️🤖

### SIH26167: Agentic Multimodal Remote-Sensing Vision-Language AI

SatQuery AI is an agentic, multimodal vision-language artificial intelligence system engineered specifically for remote-sensing imagery and geospatial natural-language interaction. Unlike generic visual chatbots, SatQuery AI dynamically inspects uploaded imagery (Optical, SAR, Bi-Temporal, GeoTIFF/TIFF), validates spatial-spectral compatibility, routes queries to specialized RS models, fuses multi-modal evidence, calculates calibrated confidence, and generates auditable execution summaries with visual evidence.

---

## 🌟 Key Features

- **Multimodal Remote Sensing Support:** Single Optical, Single SAR, Optical + SAR pairs, Bi-temporal $T_1/T_2$ pairs, and Multi-spectral GeoTIFF files.
- **Agentic Routing Engine (`AgenticRouter`):** Dynamically selects specialist tools (`SingleImageVQA`, `Captioning`, `VisualGrounding`, `ChangeDetection`, `ChangeVQA`, `OpticalSARFusion`).
- **GeoTIFF & Spatial Metadata Validation:** Full CRS, transform, resolution, band count, and spatial alignment validation.
- **Evidence-Grounded Anti-Hallucination:** Prevents language-prior hallucinations by conditioning responses strictly on visual model evidence and spatial masks.
- **Calibrated Confidence & Uncertainty:** Temperature-scaled confidence scoring with explicit uncertainty reporting.
- **Research-Grade ChatGPT Interface:** Modern Glassmorphism UI featuring interactive bounding box canvas, change-map viewer, satellite metadata inspector, and execution trace timeline.

---

## 🏗️ Architecture: QAM-RS

```text
[ USER QUERY + IMAGES ] ──► [ INPUT VALIDATOR ] ──► [ AGENTIC ROUTER ]
                                                            │
         ┌──────────────────┬─────────────────┬─────────────┴─────────────┐
         ▼                  ▼                 ▼                           ▼
  [ Single VQA ]     [ Captioner ]     [ Grounding ]        [ Change & Fusion Engine ]
         │                  │                 │                           │
         └──────────────────┴─────────────────┴───────────────────────────┘
                                       │
                                       ▼
                         [ EVIDENCE FUSION ENGINE ]
                                       │
                                       ▼
                         [ ANSWER & TRACE GENERATOR ]
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10+ (or Python 3.14 on macOS)
- PyTorch (CUDA / MPS supported)

### Environment Setup
```bash
cd satquery-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Running Backend API
```bash
python3 -m uvicorn backend.app:app --reload --port 8000
```

### Running Frontend Interface
Open `frontend/index.html` in your browser or run a simple local web server.
