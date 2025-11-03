import numpy as np
import cv2

# Đo mức nhiễu
def psnr(ref: np.ndarray, test: np.ndarray, max_val: float = 255.0) -> float:
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)
    mse = np.mean((ref - test)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

# Đo độ giống cấu trúc giữa ảnh gốc và ảnh kết quả
def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    test = test.astype(np.float64)

    K1, K2 = 0.01, 0.03
    L = 255
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    gauss = cv2.getGaussianKernel(11, 1.5)
    window = gauss @ gauss.T

    mu1 = cv2.filter2D(ref, -1, window)
    mu2 = cv2.filter2D(test, -1, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(ref * ref, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(test * test, -1, window) - mu2_sq
    sigma12   = cv2.filter2D(ref * test, -1, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())
