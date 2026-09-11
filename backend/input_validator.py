"""
SatQuery AI — Input Validator & Geospatial Metadata Engine
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Validates image formats (GeoTIFF, TIFF, PNG, JPEG), extracts CRS/transform/band metadata,
identifies image modalities (Optical, SAR, Bi-Temporal, Dual-Modality), enforces compatibility rules,
and applies sensor-aware spectral normalization without destructive modification.
"""

import os
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Suppress noisy rasterio warnings if rasterio is present
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("satquery.input_validator")


@dataclass
class ImageMetadata:
    filename: str
    filepath: str
    file_format: str
    width: int
    height: int
    num_bands: int
    dtype: str
    crs: Optional[str]
    bounds: Optional[List[float]]  # [min_x, min_y, max_x, max_y]
    transform: Optional[List[float]]
    resolution: Optional[Tuple[float, float]]  # (res_x, res_y)
    modality_type: str  # "optical_rgb", "optical_multispectral", "sar", "unknown"
    has_nan_or_inf: bool
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    spatial_resolution_rating: str = "HIGH"  # "HIGH", "MEDIUM", "COARSE"
    quality_status: str = "ACCEPTABLE"        # "ACCEPTABLE", "DEGRADED", "CORRUPTED"
    cloud_cover_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class ValidationResult:
    is_valid: bool
    num_images: int
    primary_modality: str  # "single_optical", "single_sar", "bitemporal_optical", "bitemporal_sar", "optical_sar_pair"
    image_metadata: List[ImageMetadata]
    compatibility_warnings: List[str]
    validation_errors: List[str]
    recommended_tools: List[str]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["image_metadata"] = [m.to_dict() for m in self.image_metadata]
        return res


class InputValidator:
    """
    Production Input Validator for SatQuery AI remote-sensing imagery.
    Handles GeoTIFF, TIFF, PNG, and JPEG files.
    """

    ALLOWED_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}

    def __init__(self, allow_png_jpeg: bool = True):
        self.allow_png_jpeg = allow_png_jpeg

    def extract_metadata(self, filepath: str) -> ImageMetadata:
        """Extracts complete metadata from a single geospatial or standard image."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found at path: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed: {self.ALLOWED_EXTENSIONS}")

        filename = os.path.basename(filepath)
        
        # Try GeoTIFF / Rasterio first
        if RASTERIO_AVAILABLE and ext in {".tif", ".tiff", ".geotiff"}:
            try:
                with rasterio.open(filepath) as src:
                    width = src.width
                    height = src.height
                    num_bands = src.count
                    dtype = str(src.dtypes[0])
                    crs_str = str(src.crs) if src.crs else None
                    bounds = list(src.bounds) if src.bounds else None
                    transform_matrix = list(src.transform) if src.transform else None
                    
                    res_x = abs(src.res[0]) if src.res else 0.0
                    res_y = abs(src.res[1]) if src.res else 0.0
                    resolution = (res_x, res_y)

                    # Read sample window to check statistics & NaN
                    # Read sample or full raster (limit size if large)
                    read_bands = src.read(out_shape=(num_bands, min(height, 512), min(width, 512)))
                    has_nan_inf = bool(np.isnan(read_bands).any() or np.isinf(read_bands).any())

                    # Clean data for stats calculation
                    clean_arr = np.nan_to_num(read_bands, nan=0.0, posinf=0.0, neginf=0.0)
                    min_v = float(np.min(clean_arr))
                    max_v = float(np.max(clean_arr))
                    mean_v = float(np.mean(clean_arr))
                    std_v = float(np.std(clean_arr))

                    # Estimate cloud cover (high brightness pixels)
                    cloud_pct = float(np.mean(clean_arr > (0.85 * max_v if max_v > 0 else 220)) * 100.0) if num_bands >= 3 else 0.0

                    # Estimate spatial resolution rating
                    if res_x > 0 and res_x < 1.5:
                        res_rating = "HIGH"
                    elif width * height >= 100000:
                        res_rating = "HIGH"
                    elif width * height >= 20000:
                        res_rating = "MEDIUM"
                    else:
                        res_rating = "COARSE"

                    # Quality Status
                    if has_nan_inf or std_v == 0.0:
                        quality = "CORRUPTED"
                    elif cloud_pct > 45.0:
                        quality = "DEGRADED"
                    else:
                        quality = "ACCEPTABLE"

                    modality = self._infer_single_image_modality(num_bands, min_v, max_v, std_v, filename)

                    return ImageMetadata(
                        filename=filename,
                        filepath=os.path.abspath(filepath),
                        file_format=ext[1:].upper(),
                        width=width,
                        height=height,
                        num_bands=num_bands,
                        dtype=dtype,
                        crs=crs_str,
                        bounds=bounds,
                        transform=transform_matrix,
                        resolution=resolution,
                        modality_type=modality,
                        has_nan_or_inf=has_nan_inf,
                        min_val=min_v,
                        max_val=max_v,
                        mean_val=mean_v,
                        std_val=std_v,
                        spatial_resolution_rating=res_rating,
                        quality_status=quality,
                        cloud_cover_pct=round(cloud_pct, 1)
                    )
            except Exception as e:
                logger.warning(f"Rasterio read failed for {filepath}: {e}. Falling back to standard image loader.")

        # Fallback to PIL / NumPy
        if PIL_AVAILABLE:
            with Image.open(filepath) as img:
                arr = np.array(img)
                if arr.ndim == 2:
                    num_bands = 1
                    height, width = arr.shape
                elif arr.ndim == 3:
                    height, width, num_bands = arr.shape
                else:
                    raise ValueError(f"Invalid image dimensions: {arr.shape}")

                dtype = str(arr.dtype)
                has_nan_inf = bool(np.isnan(arr).any() or np.isinf(arr).any())
                clean_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                min_v = float(np.min(clean_arr))
                max_v = float(np.max(clean_arr))
                mean_v = float(np.mean(clean_arr))
                std_v = float(np.std(clean_arr))

                # Estimate cloud cover
                if num_bands >= 3 and arr.ndim == 3:
                    r, g, b = clean_arr[:, :, 0], clean_arr[:, :, 1], clean_arr[:, :, 2]
                    cloud_mask = (r > 220) & (g > 220) & (b > 220)
                    cloud_pct = float(np.mean(cloud_mask) * 100.0)
                else:
                    cloud_pct = 0.0

                # Resolution rating
                if width * height >= 100000:
                    res_rating = "HIGH"
                elif width * height >= 20000:
                    res_rating = "MEDIUM"
                else:
                    res_rating = "COARSE"

                # Quality Status
                if has_nan_inf or std_v == 0.0:
                    quality = "CORRUPTED"
                elif cloud_pct > 45.0:
                    quality = "DEGRADED"
                else:
                    quality = "ACCEPTABLE"

                modality = self._infer_single_image_modality(num_bands, min_v, max_v, std_v, filename)

                return ImageMetadata(
                    filename=filename,
                    filepath=os.path.abspath(filepath),
                    file_format=ext[1:].upper(),
                    width=width,
                    height=height,
                    num_bands=num_bands,
                    dtype=dtype,
                    crs=None,
                    bounds=None,
                    transform=None,
                    resolution=(1.0, 1.0),
                    modality_type=modality,
                    has_nan_or_inf=has_nan_inf,
                    min_val=min_v,
                    max_val=max_v,
                    mean_val=mean_v,
                    std_val=std_v,
                    spatial_resolution_rating=res_rating,
                    quality_status=quality,
                    cloud_cover_pct=round(cloud_pct, 1)
                )
        
        raise RuntimeError("No suitable image library (Rasterio or PIL) could read the input file.")

    def _infer_single_image_modality(
        self, num_bands: int, min_v: float, max_v: float, std_v: float, filename: str
    ) -> str:
        """Helper to infer image modality (Optical RGB, Multispectral, or SAR)."""
        fn_lower = filename.lower()
        if "sar" in fn_lower or "sentinel1" in fn_lower or "vv" in fn_lower or "vh" in fn_lower or "risat" in fn_lower:
            return "sar"
        if num_bands == 1 or num_bands == 2:
            return "sar" if max_v <= 5.0 or "sar" in fn_lower else "optical_gray"
        if num_bands == 3:
            return "optical_rgb"
        if num_bands > 3:
            return "optical_multispectral"
        return "unknown"

    def validate_inputs(self, filepaths: List[str], query: str = "") -> ValidationResult:
        """
        Validates a single image or set of images for modality, compatibility, and tool recommendations.
        """
        errors = []
        warnings = []
        metadata_list = []

        if not filepaths:
            return ValidationResult(
                is_valid=False,
                num_images=0,
                primary_modality="none",
                image_metadata=[],
                compatibility_warnings=[],
                validation_errors=["No image files provided."],
                recommended_tools=[]
            )

        # 1. Extract metadata for all images
        for fp in filepaths:
            try:
                meta = self.extract_metadata(fp)
                metadata_list.append(meta)
                if meta.has_nan_or_inf:
                    warnings.append(f"Image '{meta.filename}' contains NaN/Inf values. Automatic imputation will be applied.")
            except Exception as e:
                errors.append(f"Failed to process image '{os.path.basename(fp)}': {str(e)}")

        if errors and len(metadata_list) == 0:
            return ValidationResult(
                is_valid=False,
                num_images=len(filepaths),
                primary_modality="invalid",
                image_metadata=[],
                compatibility_warnings=warnings,
                validation_errors=errors,
                recommended_tools=[]
            )

        # 2. Determine Primary Combined Modality
        num_imgs = len(metadata_list)
        modalities = [m.modality_type for m in metadata_list]
        primary_modality = "single_optical"

        if num_imgs == 1:
            if modalities[0] in {"sar"}:
                primary_modality = "single_sar"
            else:
                primary_modality = "single_optical"
        elif num_imgs == 2:
            m1, m2 = modalities[0], modalities[1]
            if (m1 in {"optical_rgb", "optical_multispectral"} and m2 == "sar") or \
               (m2 in {"optical_rgb", "optical_multispectral"} and m1 == "sar"):
                primary_modality = "optical_sar_pair"
            elif "sar" in {m1, m2}:
                primary_modality = "bitemporal_sar"
            else:
                primary_modality = "bitemporal_optical"
        else:
            primary_modality = "multitemporal_combination"

        # 3. Spatial & CRS Compatibility Check for multi-image inputs
        if num_imgs >= 2:
            m0 = metadata_list[0]
            for idx in range(1, num_imgs):
                m_curr = metadata_list[idx]
                
                # Width/Height mismatch check
                if abs(m0.width - m_curr.width) > 5 or abs(m0.height - m_curr.height) > 5:
                    warnings.append(
                        f"Image dimensions mismatch between '{m0.filename}' ({m0.width}x{m0.height}) "
                        f"and '{m_curr.filename}' ({m_curr.width}x{m_curr.height}). Resampling will be performed."
                    )
                
                # CRS Mismatch check
                if m0.crs and m_curr.crs and m0.crs != m_curr.crs:
                    warnings.append(
                        f"CRS mismatch detected: '{m0.filename}' is {m0.crs} while '{m_curr.filename}' is {m_curr.crs}. "
                        "On-the-fly reprojection will be applied."
                    )

        # 4. Recommend Tools Based on Modality & Query
        recommended_tools = self._select_recommended_tools(primary_modality, query)

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            num_images=num_imgs,
            primary_modality=primary_modality,
            image_metadata=metadata_list,
            compatibility_warnings=warnings,
            validation_errors=errors,
            recommended_tools=recommended_tools
        )

    def _select_recommended_tools(self, primary_modality: str, query: str) -> List[str]:
        """Rules engine mapping query intent + modality to agentic tools."""
        q_lower = query.lower()
        tools = []

        if "change" in q_lower or "differ" in q_lower or "increased" in q_lower or "decreased" in q_lower or "new" in q_lower:
            tools.append("change_detection")
            tools.append("change_vqa")
        elif "where" in q_lower or "locate" in q_lower or "find" in q_lower or "box" in q_lower:
            tools.append("grounding")
            tools.append("single_image_vqa")
        elif "describe" in q_lower or "caption" in q_lower or "overview" in q_lower or "summary" in q_lower:
            tools.append("captioning")
        elif "sar" in q_lower or "radar" in q_lower or primary_modality == "optical_sar_pair":
            tools.append("optical_sar_fusion")
            tools.append("single_image_vqa")
        else:
            tools.append("single_image_vqa")

        return tools

    def normalize_optical(self, array: np.ndarray, is_sentinel2_10k: bool = True) -> np.ndarray:
        """Per-band optical normalization."""
        norm_arr = array.astype(np.float32)
        if is_sentinel2_10k and np.max(norm_arr) > 1.0:
            norm_arr = norm_arr / 10000.0
            norm_arr = np.clip(norm_arr, 0.0, 1.0)
        else:
            min_v, max_v = np.min(norm_arr), np.max(norm_arr)
            if max_v > min_v:
                norm_arr = (norm_arr - min_v) / (max_v - min_v)
        return norm_arr

    def normalize_sar(self, array: np.ndarray, to_decibel: bool = True) -> np.ndarray:
        """Sensor-specific SAR normalization (linear amplitude -> dB scale [-30, 0] dB -> [0, 1])."""
        sar_arr = array.astype(np.float32)
        if to_decibel and np.min(sar_arr) >= 0.0:
            # Prevent log(0)
            eps = 1e-6
            sar_db = 10.0 * np.log10(np.maximum(sar_arr, eps))
            # Clip between -30 dB (noise floor) and 0 dB (bright target)
            norm_sar = np.clip((sar_db + 30.0) / 30.0, 0.0, 1.0)
            return norm_sar
        else:
            min_v, max_v = np.min(sar_arr), np.max(sar_arr)
            if max_v > min_v:
                return (sar_arr - min_v) / (max_v - min_v)
            return sar_arr
