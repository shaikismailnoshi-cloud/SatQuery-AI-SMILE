import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def create_dense_residential_image(filepath="/tmp/dense_residential_200x185.png"):
    """
    Creates a synthetic 200x185 satellite image of a dense residential neighborhood with ~24 individual houses.
    Grid of 6 columns x 4 rows of neighboring houses with small gaps.
    """
    w, h = 200, 185
    arr = np.ones((h, w, 3), dtype=np.uint8) * 90 # Asphalt/ground background

    # Green lawns / trees between blocks
    arr[10:175, 5:15] = [40, 120, 40]
    arr[10:175, 185:195] = [40, 120, 40]

    count = 0
    # 6 columns, 4 rows of houses
    for row in range(4):
        y_start = 20 + row * 38
        y_end = y_start + 28
        for col in range(6):
            x_start = 25 + col * 26
            x_end = x_start + 20

            # Alternate roof colors: Red tiles, Blue tin, White metal, Brown shingles
            color_idx = (row + col) % 4
            if color_idx == 0:
                color = [200, 60, 50]    # Red tile roof
            elif color_idx == 1:
                color = [50, 100, 190]   # Blue tin roof
            elif color_idx == 2:
                color = [225, 225, 230]  # White metal roof
            else:
                color = [140, 100, 70]   # Brown shingles

            arr[y_start:y_end, x_start:x_end] = color
            # Draw dark ridge line across roof center
            arr[y_start + 14, x_start:x_end] = [30, 30, 30]
            count += 1

    img = Image.fromarray(arr)
    img.save(filepath, format="PNG")
    print(f"Created synthetic dense residential image: {filepath} ({w}x{h}) with {count} distinct houses.")
    return filepath

if __name__ == "__main__":
    create_dense_residential_image()
