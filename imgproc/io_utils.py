import os
import cv2
from typing import List

# Tạo thư mục nếu chưa tồn tại
def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

# Kiểm tra file có phải ảnh không
def is_image_file(p: str) -> bool:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    return p.lower().endswith(exts)

# Liệt kê tất cả ảnh trong thư mục
def list_images(folder: str) -> List[str]:
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if is_image_file(f)]

# Đọc ảnh ở dạng đen trắng
def load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

# Lưu ảnh kết quả ra file
def save_image(path: str, img):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)
