import os
import argparse
import cv2

from imgproc.io_utils import ensure_dir, load_gray, save_image, list_images, is_image_file
from imgproc.visualize import save_grid
from imgproc.pipeline import PipelineConfig, run_pipeline

def process_one_image(img_path: str, outdir: str, cfg: PipelineConfig):
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_gray(img_path)

    # chạy pipeline
    result = run_pipeline(img, cfg)

    vi_names = {
        "clahe": "Tang_tuong_phan_CLAHE",
        "he": "Tang_tuong_phan_HE",
        "denoise_median": "Loc_nhieu_Median",
        "denoise_gauss": "Loc_nhieu_Gaussian",
        "highboost": "Lam_sac_HighBoost",
        "unsharp": "Lam_sac_Unsharp",
        "edges_canny": "Bien_Canny",
        "edges_sobel": "Bien_Sobel",
        "edges_laplacian": "Bien_Laplacian",
    }

    step_imgs = [("Anh_goc", img)]
    for k, v in result["steps"].items():
        vi_name = vi_names.get(k, k)
        out_p = os.path.join(outdir, name, f"{vi_name}.png")
        save_image(out_p, v)
        step_imgs.append((vi_name, v))

    grid_path = os.path.join(outdir, name, "Tong_quan_qua_trinh.png")
    save_grid(grid_path, step_imgs)

    print(f"\n== Metrics for {name} ==")
    for k, m in result["metrics"].items():
        print(f"{k:20s}  PSNR={m['psnr_vs_input']:.2f}  SSIM={m['ssim_vs_input']:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Image Toolkit – Capstone")
    ap.add_argument("--input", required=True, help="Đường dẫn ảnh hoặc thư mục ảnh")
    ap.add_argument("--outdir", default="outputs", help="Thư mục xuất")
    ap.add_argument("--batch", action="store_true", help="Chế độ xử lý cả thư mục")
    ap.add_argument("--he", action="store_true", help="Dùng Histogram Equalization")
    ap.add_argument("--clahe", action="store_true", help="Dùng CLAHE")
    ap.add_argument("--mean", type=int, default=0, help="Kernel mean (0=off)")
    ap.add_argument("--median", type=int, default=0, help="Kernel median (0=off)")
    ap.add_argument("--gauss", type=int, default=0, help="Kernel gaussian (0=off)")
    ap.add_argument("--unsharp", type=float, default=0.0, help="Unsharp amount (0=off)")
    ap.add_argument("--highboost", type=float, default=0.0, help="High-boost A (0=off)")
    ap.add_argument("--edges", nargs="*", default=["sobel","laplacian","canny"],
                    help="Các phương pháp biên: sobel laplacian canny")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    cfg = PipelineConfig(
        use_he=args.he,
        use_clahe=args.clahe,
        mean_ksize=(args.mean or None),
        median_ksize=(args.median or None),
        gauss_ksize=(args.gauss or None),
        unsharp_amount=(args.unsharp if args.unsharp > 0 else None),
        highboost_A=(args.highboost if args.highboost > 0 else None),
        edge_methods=args.edges
    )

    if args.batch:
        if not os.path.isdir(args.input):
            raise ValueError("Ở chế độ --batch, --input phải là thư mục!")
        imgs = [p for p in list_images(args.input)]
        if not imgs:
            print("Không tìm thấy ảnh trong thư mục.")
            return
        for p in imgs:
            print(f"Processing: {p}")
            process_one_image(p, args.outdir, cfg)
    else:
        if not is_image_file(args.input):
            raise ValueError("Hãy chỉ định đường dẫn tới 1 ảnh hợp lệ, hoặc dùng --batch cho thư mục.")
        process_one_image(args.input, args.outdir, cfg)

if __name__ == "__main__":
    main()
