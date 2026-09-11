# SIH26167 Compliance & Feature Verification Document

## SatQuery AI: Agentic Multimodal Remote-Sensing Vision-Language AI System

---

### Requirement Mapping Matrix

| # | SIH26167 Requirement | Technical Implementation | Dataset / Baseline | Verification & Evidence | Status |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **1** | **Remote-Sensing VLM Adaptation** | RS-adapted Vision Encoder + Multispectral/SAR projector fine-tuned via `BigEarthNet.txt` | `BigEarthNet.txt` (Sentinel-1/2) | Visual representation adaptation loss curves & evaluation metrics in `reports/` | **VERIFIED** |
| **2** | **Single-Image VQA** | `SingleImageVQA_Specialist` with Query-Conditioned Cross-Attention | `VRSBench` & `RSVQA` | VQA accuracy, F1, exact match metrics on held-out test splits | **VERIFIED** |
| **3** | **Captioning & Visual Grounding** | `Captioner` (spatial description) + `GroundingEngine` (BBox `[ymin, xmin, ymax, xmax]`) | `VRSBench` | BLEU/CIDEr caption scores & Bounding Box overlay canvas on UI | **VERIFIED** |
| **4** | **Multitemporal Change Understanding** | Siamese Temporal Encoder $T_1+T_2+\Delta(T_1, T_2)$ + Change Mask Generator & Change VQA | `CDVQA`, `LEVIR-CD`, `WHU-CD` | Pixel-level Change Detection Map + Natural Language Change Description | **VERIFIED** |
| **5** | **Optical-SAR Analysis** | Dual-Stream Gated Cross-Modal Encoder (Optical RGB/Multispectral + SAR VV/VH decibel scaling) | `SEN12MS` & `BigEarthNet-v2.0` | Complementary Modality Fusion matrix & Ablation report | **VERIFIED** |
| **6** | **Agentic Orchestration** | Dynamic `AgenticRouter` parsing query semantics, file count, CRS, modality, & resolution | Custom Rule + LLM Agentic Planner | Execution summary trace logging tools, arguments, and confidence | **VERIFIED** |
| **7** | **GeoTIFF / TIFF Validation** | `InputValidator` with GDAL/Rasterio CRS, bounds, transform, and per-band normalization | Cartosat-2S & RISAT SAR compatible pipeline | Georeferencing metadata preservation & spatial alignment error reporting | **VERIFIED** |
| **8** | **Evidence-Grounded Anti-Hallucination** | `EvidenceFusionEngine` supplying structured model predictions & spatial masks to final answer generator | Cross-Model Consensus Engine | Return `"Insufficient visual evidence..."` when evidence is ambiguous | **VERIFIED** |
| **9** | **Calibrated Confidence & Uncertainty** | Temperature-scaled confidence calculator & uncertainty score output | Calibrated probability distribution | Numerical confidence score + High/Medium/Low qualitative badge | **VERIFIED** |
| **10** | **Auditable Execution Trace** | Step-by-step tool execution summary output in API response payload & UI timeline | `AgenticRouter` trace collector | Full auditable pipeline trace visible in UI panel without exposing CoT | **VERIFIED** |
| **11** | **Research-Grade Modern UI** | ChatGPT-like interface built with glassmorphism aesthetics, drag-drop GeoTIFF upload, canvas viewer | Custom CSS/JS Frontend | Real-time multi-image upload, change map visualization, metadata inspect | **VERIFIED** |

---

### Target Sensor & Generalization Policy
* **SIH / ISRO Target Compatibility:** Designed to natively ingest **Cartosat-2S** (high-resolution optical), **RISAT SAR** (C-band SAR), Sentinel-1/2, and generic GeoTIFF/TIFF imagery.
* **Georeferenced Resampling:** All reprojections/resamplings preserve original affine transformation matrices and coordinate reference systems (CRS) without destructive pixel alteration.
