import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .io_utils import ensure_dir

# Tạo figure n x 3, hiển thị ảnh theo thứ tự
def save_grid(out_path: str, titles_imgs):
    n = len(titles_imgs)
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4*cols, 3.5*rows))
    for i, (title, im) in enumerate(titles_imgs, 1):
        plt.subplot(rows, cols, i)
        cmap = 'gray' if (len(im.shape) == 2) else None
        plt.imshow(im, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    ensure_dir(os.path.dirname(out_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
