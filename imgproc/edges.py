import cv2
import numpy as np

# Phát hiện biên theo hướng X-Y sử dụng Sobel
def sobel_edges(img_gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag

# Phát hiện biên tổng quát (Laplacian)
def laplacian_edges(img_gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
    lap = np.absolute(lap)
    lap = np.clip(lap, 0, 255).astype(np.uint8)
    return lap

# Phát hiện biên hiệu quả, lọc nhiễu tốt (Canny)
def canny_edges(img_gray: np.ndarray, t1=100, t2=200) -> np.ndarray:
    return cv2.Canny(img_gray, t1, t2)
