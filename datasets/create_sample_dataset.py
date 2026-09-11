"""
SatQuery AI — Synthetic & Benchmark Sample Imagery Generator
SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

Generates sample geospatial GeoTIFF and TIFF files representing:
1. Single Optical RGB (Cartosat-2S style)
2. Single SAR (RISAT / Sentinel-1 style amplitude & dB)
3. Bi-Temporal Optical Pair (T1 before, T2 after with synthetic building & vegetation changes)
4. Co-registered Optical + SAR Pair
"""

import os
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_AVAIL = True
except ImportError:
    RASTERIO_AVAIL = False

SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_imagery"))
os.makedirs(SAMPLE_DIR, exist_ok=True)


def create_optical_geotiff(filename: str, width: int = 512, height: int = 512):
    """Generates a 3-band Optical RGB GeoTIFF with geospatial metadata."""
    filepath = os.path.join(SAMPLE_DIR, filename)
    
    # Synthetic land cover pattern (green vegetation, blue water body, grey urban structures)
    data = np.zeros((3, height, width), dtype=np.uint8)
    
    # Background vegetation (Green channel high)
    data[0, :, :] = np.random.randint(40, 80, (height, width))
    data[1, :, :] = np.random.randint(120, 200, (height, width))
    data[2, :, :] = np.random.randint(30, 70, (height, width))

    # Water stream (Blue channel high)
    data[0, 100:150, :] = 20
    data[1, 100:150, :] = 80
    data[2, 100:150, :] = 220

    # Building clusters (RGB equal high)
    data[:, 250:350, 200:300] = np.random.randint(180, 240, (3, 100, 100))

    if RASTERIO_AVAIL:
        transform = from_origin(77.5946, 12.9716, 0.0001, 0.0001)  # WGS84 coordinates near Bangalore
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(data)
    else:
        # Fallback to PNG/TIFF via PIL
        img = Image.fromarray(np.transpose(data, (1, 2, 0)))
        img.save(filepath)

    print(f"Created Optical Sample: {filepath}")
    return filepath


def create_sar_geotiff(filename: str, width: int = 512, height: int = 512):
    """Generates a Single-Band SAR (VV Polarization) GeoTIFF."""
    filepath = os.path.join(SAMPLE_DIR, filename)
    
    # SAR Backscatter amplitude (urban high specular reflection, water smooth low reflection)
    sar_data = np.random.normal(15.0, 5.0, (1, height, width)).astype(np.float32)
    
    # Water stream (very low backscatter)
    sar_data[0, 100:150, :] = np.random.normal(2.0, 0.5, (50, width))
    
    # Urban structures (high double-bounce backscatter)
    sar_data[0, 250:350, 200:300] = np.random.normal(85.0, 15.0, (100, 100))

    sar_data = np.clip(sar_data, 0.1, 255.0)

    if RASTERIO_AVAIL:
        transform = from_origin(77.5946, 12.9716, 0.0001, 0.0001)
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(sar_data)
    else:
        img = Image.fromarray(sar_data[0].astype(np.uint8))
        img.save(filepath)

    print(f"Created SAR Sample: {filepath}")
    return filepath


def create_bitemporal_pair():
    """Generates T1 (before) and T2 (after) bi-temporal optical GeoTIFF pairs with synthetic changes."""
    t1_path = create_optical_geotiff("bitemporal_t1_before.tif")
    t2_filepath = os.path.join(SAMPLE_DIR, "bitemporal_t2_after.tif")
    
    # Read T1 data and introduce a new building and vegetation clearing in T2
    if RASTERIO_AVAIL:
        with rasterio.open(t1_path) as src:
            t2_data = src.read()
            profile = src.profile

        # New construction in top-right quadrant
        t2_data[:, 50:150, 350:450] = np.random.randint(200, 250, (3, 100, 100))
        # Vegetation loss in bottom-left
        t2_data[1, 380:480, 50:180] = 60  # Reduced greenness

        with rasterio.open(t2_filepath, 'w', **profile) as dst:
            dst.write(t2_data)
    else:
        img_t1 = Image.open(t1_path)
        arr = np.array(img_t1)
        arr[50:150, 350:450] = 220
        arr[380:480, 50:180, 1] = 60
        Image.fromarray(arr).save(t2_filepath)

    print(f"Created Bi-Temporal T2 Sample: {t2_filepath}")
    return t1_path, t2_filepath


if __name__ == "__main__":
    print("Generating SatQuery AI sample satellite dataset...")
    create_optical_geotiff("cartosat2s_optical_sample.tif")
    create_sar_geotiff("risat_sar_sample.tif")
    create_bitemporal_pair()
    print("Sample dataset generation complete! Files saved in 'datasets/sample_imagery/'.")
