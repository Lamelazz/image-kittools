import cv2
import numpy as np
from typing import List

# Lọc trung bình (Mean Filter) – làm mịn ảnh
def mean_filter(img_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = (ksize, ksize)
    return cv2.blur(img_gray, k)

# Lọc trung vị (Median Filter) – xóa nhiễu muối tiêu
def median_filter(img_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img_gray, ksize)

# Lọc Gaussian – giảm nhiễu, làm mượt ảnh
def gaussian_filter(img_gray: np.ndarray, ksize: int = 3, sigma: float = 0) -> np.ndarray:
    k = (ksize, ksize)
    return cv2.GaussianBlur(img_gray, k, sigmaX=sigma)

# Trung bình nhiều khung hình – giảm nhiễu trong video
def multi_frame_average(frames: List[np.ndarray]) -> np.ndarray:
    acc = np.zeros_like(frames[0], dtype=np.float64)
    for f in frames:
        acc += f.astype(np.float64)
    avg = (acc / len(frames)).round().astype(np.uint8)
    return avg
