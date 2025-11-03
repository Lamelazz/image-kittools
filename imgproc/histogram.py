import cv2
import numpy as np

# Tăng tương phản toàn ảnh
def hist_equalize(img_gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img_gray)

# CLAHE (Contrast Limited Adaptive HE) – tăng tương phản cục bộ, tránh cháy sáng
def clahe(img_gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8,8)) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return c.apply(img_gray)
