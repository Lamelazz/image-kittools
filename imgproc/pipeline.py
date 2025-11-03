from dataclasses import dataclass
from email.mime import base
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from . import histogram as H
from . import denoise as D
from . import edges as E
from . import sharpen as S
from . import metrics as M

@dataclass
class PipelineConfig:
    use_he: bool = True
    use_clahe: bool = False
    mean_ksize: Optional[int] = None
    median_ksize: Optional[int] = None
    gauss_ksize: Optional[int] = None
    unsharp_amount: Optional[float] = None
    highboost_A: Optional[float] = None
    edge_methods: List[str] = None  # ["sobel","laplacian","canny"]

def run_pipeline(img_gray: np.ndarray, cfg: PipelineConfig) -> Dict[str, Any]:
    out = {"steps": {}}

    base = img_gray.copy()
    if cfg.use_he:
        base = H.hist_equalize(base)
        out["steps"]["he"] = base
    if cfg.use_clahe:
        base = H.clahe(base)
        out["steps"]["clahe"] = base

    denoised = {}
    if cfg.mean_ksize:
        denoised["mean"] = D.mean_filter(base, cfg.mean_ksize)
    if cfg.median_ksize:
        denoised["median"] = D.median_filter(base, cfg.median_ksize)
    if cfg.gauss_ksize:
        denoised["gauss"] = D.gaussian_filter(base, cfg.gauss_ksize)

    if "median" in denoised:
        current = denoised["median"]
    elif "gauss" in denoised:
        current = denoised["gauss"]
    elif "mean" in denoised:
        current = denoised["mean"]
    else:
        current = base
    out["steps"].update({f"denoise_{k}": v for k, v in denoised.items()})

    if cfg.unsharp_amount:
        current = S.unsharp_mask(current, amount=cfg.unsharp_amount)
        out["steps"]["unsharp"] = current
    if cfg.highboost_A:
        current = S.high_boost(current, A=cfg.highboost_A)
        out["steps"]["highboost"] = current

    edges = {}
    methods = cfg.edge_methods or []
    for m in methods:
        if m == "sobel":
            edges["sobel"] = E.sobel_edges(current)
        elif m == "laplacian":
            edges["laplacian"] = E.laplacian_edges(current)
        elif m == "canny":
            edges["canny"] = E.canny_edges(current)
    out["steps"].update({f"edges_{k}": v for k, v in edges.items()})

    out["metrics"] = {}
    for name, img in out["steps"].items():
        out["metrics"][name] = {
            "psnr_vs_input": M.psnr(img_gray, img),
            "ssim_vs_input": M.ssim(img_gray, img)
        }
    return out
