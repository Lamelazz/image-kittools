import cv2
import numpy as np

def _to_float(img_gray: np.ndarray) -> np.ndarray:
    return img_gray.astype(np.float32)

# Làm sắc bằng cách trừ ảnh mờ
def unsharp_mask(img_gray: np.ndarray, amount: float = 1.0, blur_ksize: int = 5) -> np.ndarray:
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    img32   = _to_float(img_gray)
    blurred = cv2.GaussianBlur(img32, (blur_ksize, blur_ksize), 0)
    detail  = img32 - blurred
    sharp32 = img32 + amount * detail
    return np.clip(sharp32, 0, 255).astype(np.uint8)

# Làm sắc mạnh hơn, khuếch đại biên độ
def high_boost(img_gray: np.ndarray, A: float = 1.5, blur_ksize: int = 5) -> np.ndarray:
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    img32   = _to_float(img_gray)
    blurred = cv2.GaussianBlur(img32, (blur_ksize, blur_ksize), 0)
    mask    = img32 - blurred
    boost32 = img32 + A * mask
    return np.clip(boost32, 0, 255).astype(np.uint8)
