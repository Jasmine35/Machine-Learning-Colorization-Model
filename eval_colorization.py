# Converts each color image to the model input (L channel)
# Runs the model to predict ab channels 
# Reconstructs predicted color image
# Computes PSNR, SSIM, and mean ΔE2000 (Lab)
# Builds a chromatic confusion heatmap by quantizing ab space
# Saves metrics, graphs, and heatmaps

import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import torch
from train_colorization import ColorizationNetWithResNet  # Updated to your model class
from skimage.color import deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# tools for model loading and image processing
def load_model(weights_path, device="cpu"):
    """Load model; supports checkpoint dicts with 'model_state' or direct state_dicts."""
    model = ColorizationNetWithResNet()  # Updated to your model class
    map_location = device if isinstance(device, torch.device) else torch.device(device)
    ckpt = torch.load(weights_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.to(map_location)
    model.eval()
    print(f"Model loaded from {weights_path}")
    return model

# image processing and metric computation functions
def preprocess_image_to_L(img_bgr, rs=(256,256)):
    """Return L normalized (H,W) in [0,1], and lab_resized (uint8) for GT"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab_rs = cv2.resize(lab, rs)
    L = lab_rs[:, :, 0].astype(np.float32) / 255.0
    return L, lab_rs

# reconstruct lab image from L and predicted ab
def reconstruct_lab_from_pred(L_norm, ab_norm):
    """
    L_norm: (H,W) in [0,1]
    ab_norm: (H,W,2) in normalized [-1,1] (model output)
    Returns lab_out uint8 in OpenCV LAB space (L:0-255, a,b:0-255)
    """
    L_uint8 = (L_norm * 255.0).clip(0,255).astype(np.uint8)
    ab_255 = (ab_norm * 128.0 + 128.0).clip(0,255).astype(np.uint8)
    lab_out = np.zeros((L_uint8.shape[0], L_uint8.shape[1], 3), dtype=np.uint8)
    lab_out[:,:,0] = L_uint8
    lab_out[:,:,1:] = ab_255
    return lab_out

# utility functions for metrics
def lab_to_rgb_for_metrics(lab_opencv_uint8):
    """Convert OpenCV LAB uint8 -> RGB float [0,1]"""
    bgr = cv2.cvtColor(lab_opencv_uint8, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = (rgb.astype(np.float32) / 255.0).clip(0,1)
    return rgb

# compute per-image metrics
def lab_arrays_for_deltaE(lab_opencv_uint8, L_from_norm=None):
    """Return Lab array in scikit-image expected scale: L [0,100], a/b centered around 0"""
    lab = lab_opencv_uint8.astype(np.float32)
    if L_from_norm is None:
        L = lab[:,:,0] * (100.0/255.0)
    else:
        L = (L_from_norm * 100.0).astype(np.float32)
    a = lab[:,:,1].astype(np.float32) - 128.0
    b = lab[:,:,2].astype(np.float32) - 128.0
    return np.dstack([L, a, b])

# compute metrics for one image
def compute_metrics_per_image(gt_bgr, pred_lab_uint8, L_norm):
    """Compute PSNR, SSIM, mean ΔE2000 for one image."""
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pred_rgb = lab_to_rgb_for_metrics(pred_lab_uint8)

    try:
        psnr = sk_psnr(gt_rgb, pred_rgb, data_range=1.0)
    except Exception:
        psnr = float("nan")

    try:
        ssim = sk_ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=1.0)
    except Exception:
        ssim = float("nan")

    # GT lab resized
    gt_lab_rs = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2LAB)
    gt_lab_rs = cv2.resize(gt_lab_rs, (pred_lab_uint8.shape[1], pred_lab_uint8.shape[0]))
    lab_gt = lab_arrays_for_deltaE(gt_lab_rs, L_from_norm=None)
    lab_pred = lab_arrays_for_deltaE(pred_lab_uint8, L_from_norm=L_norm)

    try:
        deltaE_map = deltaE_ciede2000(lab_gt, lab_pred)
        mean_dE = float(np.nanmean(deltaE_map))
        std_dE = float(np.nanstd(deltaE_map))
    except Exception:
        mean_dE = float("nan")
        std_dE = float("nan")

    return psnr, ssim, mean_dE, std_dE

# chromatic confusion matrix functions
def build_chromatic_confusion_matrix(accum_counts, n_bins=32):
    """Normalize rows to probabilities."""
    counts = accum_counts.astype(np.float32)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = np.nan_to_num(counts / (row_sums + 1e-8))
    return probs

# quantize ab channels to bins
def quantize_ab_to_bins(ab_centered, n_bins=32, a_range=(-128,127), b_range=(-128,127)):
    """Quantize centered a/b into bin indices in [0, n_bins-1]."""
    a = ab_centered[:,:,0]
    b = ab_centered[:,:,1]
    a_min, a_max = a_range
    b_min, b_max = b_range
    a_idx = np.floor((a - a_min) / (a_max - a_min + 1e-8) * n_bins).astype(int)
    b_idx = np.floor((b - b_min) / (b_max - b_min + 1e-8) * n_bins).astype(int)
    a_idx = np.clip(a_idx, 0, n_bins-1)
    b_idx = np.clip(b_idx, 0, n_bins-1)
    return a_idx, b_idx

# main evaluation loop - evaluate all images in folder
def evaluate_folder(model, data_folder, out_dir, device="cpu", n_bins=32, save_individual_predictions=False):
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(glob(os.path.join(data_folder, "*.jpg")) +
                       glob(os.path.join(data_folder, "*.jpeg")) +
                       glob(os.path.join(data_folder, "*.png")))

    results = []
    N = n_bins * n_bins
    confusion_counts = np.zeros((N, N), dtype=np.int64)
    
    # Store per-image deltaE maps for overall histogram
    all_deltaE_maps = []
    
    # For output color bias analysis
    output_stats = {"a_mean": [], "b_mean": [], "a_std": [], "b_std": []}

    # Ensure model is on correct device
    dev = device if isinstance(device, torch.device) else torch.device(device)
    model.to(dev)
    model.eval()

    for p in tqdm(img_paths, desc="Evaluating images"):
        fname = os.path.basename(p)
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            print("Failed to read", p)
            continue

        # Preprocess
        L_norm, gt_lab_rs = preprocess_image_to_L(img_bgr, rs=(256,256))

        # Build input tensor
        L_tensor = torch.tensor(L_norm).unsqueeze(0).unsqueeze(0).float().to(dev)

        # Forward
        with torch.no_grad():
            out = model(L_tensor).cpu().numpy()
        pred_ab = out[0].transpose(1,2,0)  # (H,W,2) in [-1,1]

        # Reconstruct lab and compute metrics
        pred_lab_uint8 = reconstruct_lab_from_pred(L_norm, pred_ab)
        psnr, ssim, mean_dE, std_dE = compute_metrics_per_image(img_bgr, pred_lab_uint8, L_norm)

        # Collect output statistics
        output_stats["a_mean"].append(np.mean(pred_ab[:,:,0]))
        output_stats["b_mean"].append(np.mean(pred_ab[:,:,1]))
        output_stats["a_std"].append(np.std(pred_ab[:,:,0]))
        output_stats["b_std"].append(np.std(pred_ab[:,:,1]))
        
        results.append({
            "filename": fname, 
            "psnr": psnr, 
            "ssim": ssim, 
            "deltaE2000_mean": mean_dE,
            "deltaE2000_std": std_dE
        })

        # Save individual prediction if requested
        if save_individual_predictions:
            pred_rgb = lab_to_rgb_for_metrics(pred_lab_uint8)
            pred_bgr = cv2.cvtColor((pred_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"pred_{fname}"), pred_bgr)

        # --- chromatic confusion accumulation ---
        gt_ab_centered = (gt_lab_rs[:,:,1:3].astype(np.float32) - 128.0)  # (H,W,2)
        pred_ab_centered = pred_ab * 128.0  # (H,W,2)

        gt_a_idx, gt_b_idx = quantize_ab_to_bins(gt_ab_centered, n_bins=n_bins)
        pr_a_idx, pr_b_idx = quantize_ab_to_bins(pred_ab_centered, n_bins=n_bins)

        gt_flat = (gt_a_idx * n_bins + gt_b_idx).ravel()
        pr_flat = (pr_a_idx * n_bins + pr_b_idx).ravel()

        # accumulate counts
        np.add.at(confusion_counts, (gt_flat, pr_flat), 1)
        
        # Store deltaE map for overall histogram
        try:
            gt_lab_rs_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            gt_lab_rs_full = cv2.resize(gt_lab_rs_full, (pred_lab_uint8.shape[1], pred_lab_uint8.shape[0]))
            lab_gt = lab_arrays_for_deltaE(gt_lab_rs_full, L_from_norm=None)
            lab_pred = lab_arrays_for_deltaE(pred_lab_uint8, L_from_norm=L_norm)
            deltaE_map = deltaE_ciede2000(lab_gt, lab_pred)
            all_deltaE_maps.append(deltaE_map)
        except:
            pass

    # Save per-image CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "per_image_metrics.csv"), index=False)

    # Summary statistics
    avg_psnr = float(df["psnr"].mean())
    avg_ssim = float(df["ssim"].mean())
    avg_deltaE = float(df["deltaE2000_mean"].mean())
    std_deltaE = float(df["deltaE2000_mean"].std())
    
    # Output color bias statistics
    avg_a_mean = np.mean(output_stats["a_mean"])
    avg_b_mean = np.mean(output_stats["b_mean"])
    avg_a_std = np.mean(output_stats["a_std"])
    avg_b_std = np.mean(output_stats["b_std"])
    
    summary = {
        "avg_psnr": avg_psnr, 
        "avg_ssim": avg_ssim, 
        "avg_deltaE2000": avg_deltaE,
        "std_deltaE2000": std_deltaE,
        "avg_a_mean": avg_a_mean,
        "avg_b_mean": avg_b_mean,
        "avg_a_std": avg_a_std,
        "avg_b_std": avg_b_std,
        "n_images": len(df)
    }
    
    pd.Series(summary).to_csv(os.path.join(out_dir, "summary_metrics.csv"))
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Number of images: {len(df)}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average ΔE2000: {avg_deltaE:.2f} (std: {std_deltaE:.2f})")
    print(f"Output color bias - a channel: {avg_a_mean:.3f} (std: {avg_a_std:.3f})")
    print(f"Output color bias - b channel: {avg_b_mean:.3f} (std: {avg_b_std:.3f})")
    print("="*60)

    # Create comprehensive visualization plots
    create_visualization_plots(df, all_deltaE_maps, confusion_counts, output_stats, out_dir, n_bins)
    
    return summary, df

def create_visualization_plots(df, all_deltaE_maps, confusion_counts, output_stats, out_dir, n_bins):
    """Create all visualization plots."""
    
    # 1. Metrics distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSNR histogram
    axes[0,0].hist(df["psnr"].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0,0].axvline(df["psnr"].mean(), color='red', linestyle='--', label=f'Mean: {df["psnr"].mean():.2f}')
    axes[0,0].set_xlabel('PSNR (dB)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('PSNR Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # SSIM histogram
    axes[0,1].hist(df["ssim"].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0,1].axvline(df["ssim"].mean(), color='red', linestyle='--', label=f'Mean: {df["ssim"].mean():.3f}')
    axes[0,1].set_xlabel('SSIM')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('SSIM Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # DeltaE histogram
    axes[1,0].hist(df["deltaE2000_mean"].dropna(), bins=40, edgecolor='black', alpha=0.7)
    axes[1,0].axvline(df["deltaE2000_mean"].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["deltaE2000_mean"].mean():.2f}')
    axes[1,0].set_xlabel('ΔE2000')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('ΔE2000 Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Scatter plot: PSNR vs DeltaE
    axes[1,1].scatter(df["deltaE2000_mean"], df["psnr"], alpha=0.6, edgecolors='w', s=50)
    axes[1,1].set_xlabel('ΔE2000')
    axes[1,1].set_ylabel('PSNR (dB)')
    axes[1,1].set_title('PSNR vs ΔE2000 Correlation')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_distributions.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Output color bias analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # a-channel distribution
    axes[0].hist(output_stats["a_mean"], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0].axvline(np.mean(output_stats["a_mean"]), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(output_stats["a_mean"]):.3f}')
    axes[0].set_xlabel('a-channel mean (normalized)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('a-channel Bias Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # b-channel distribution
    axes[1].hist(output_stats["b_mean"], bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[1].axvline(np.mean(output_stats["b_mean"]), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(output_stats["b_mean"]):.3f}')
    axes[1].set_xlabel('b-channel mean (normalized)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('b-channel Bias Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "color_bias_analysis.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Chromatic confusion heatmaps
    # Build marginal matrix
    marg = np.zeros((n_bins, n_bins), dtype=np.float32)
    for a_t in range(n_bins):
        for a_p in range(n_bins):
            row_start = a_t * n_bins
            row_end = (a_t+1) * n_bins
            col_start = a_p * n_bins
            col_end = (a_p+1) * n_bins
            marg[a_t, a_p] = confusion_counts[row_start:row_end, col_start:col_end].sum()
    
    # Normalize
    marg_norm = marg / (marg.sum() + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    im = axes[0].imshow(marg_norm, origin='lower', cmap='inferno', aspect='auto')
    axes[0].set_xlabel('Predicted a-bin')
    axes[0].set_ylabel('True a-bin')
    axes[0].set_title('Chromatic Confusion Matrix (a-channel)')
    plt.colorbar(im, ax=axes[0], label='Normalized frequency')
    
    # Histogram of diagonal vs off-diagonal
    diag_vals = np.diag(marg_norm)
    off_diag_vals = marg_norm[np.eye(n_bins, dtype=bool) == False]
    
    axes[1].hist(diag_vals, bins=30, alpha=0.7, label='Diagonal (correct)', color='green')
    axes[1].hist(off_diag_vals, bins=30, alpha=0.5, label='Off-diagonal (errors)', color='red')
    axes[1].set_xlabel('Normalized frequency')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confusion Matrix Element Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chromatic_confusion_analysis.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 4. Overall deltaE map (aggregate)
    if all_deltaE_maps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Flatten all deltaE maps
        all_deltaE_flat = np.concatenate([d.flatten() for d in all_deltaE_maps])
        
        # Histogram with percentile markers
        axes[0].hist(all_deltaE_flat, bins=100, edgecolor='black', alpha=0.7)
        for p in [50, 75, 90, 95]:
            percentile = np.percentile(all_deltaE_flat, p)
            axes[0].axvline(percentile, color='red', linestyle='--', alpha=0.7, 
                           label=f'{p}th: {percentile:.1f}')
        axes[0].set_xlabel('ΔE2000')
        axes[0].set_ylabel('Pixel Count (log)')
        axes[0].set_yscale('log')
        axes[0].set_title('Aggregate ΔE2000 Distribution (all pixels)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # CDF plot
        sorted_dE = np.sort(all_deltaE_flat)
        cdf = np.arange(1, len(sorted_dE)+1) / len(sorted_dE)
        axes[1].plot(sorted_dE, cdf, linewidth=2)
        axes[1].set_xlabel('ΔE2000')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('CDF of ΔE2000')
        axes[1].grid(True, alpha=0.3)
        
        # Add some useful percentile lines
        for p in [50, 75, 90, 95]:
            percentile = np.percentile(all_deltaE_flat, p)
            axes[1].axvline(percentile, color='red', linestyle='--', alpha=0.5)
            axes[1].text(percentile, 0.1, f'{p}%', rotation=90, va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "aggregate_deltaE_analysis.png"), dpi=200, bbox_inches='tight')
        plt.close()

def main():
    # argument parsing
    parser = argparse.ArgumentParser(description="Evaluate colorization model metrics")
    parser.add_argument("--model", type=str, default="colorization_model.pth", 
                       help="Path to model weights or checkpoint")
    parser.add_argument("--data_folder", type=str, default="test_samples", 
                       help="Folder containing test images")
    parser.add_argument("--out_dir", type=str, default="eval_out", 
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","mps","cuda"],
                       help="Device to run evaluation on")
    parser.add_argument("--n_bins", type=int, default=24,
                       help="Number of bins for chromatic confusion matrix")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save individual predicted images")
    args = parser.parse_args()

    # determine device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    model = load_model(args.model, device=device)
    summary, df = evaluate_folder(model, args.data_folder, args.out_dir, 
                                 device=device, n_bins=args.n_bins,
                                 save_individual_predictions=args.save_predictions)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.out_dir}")
    print(f"Key metrics CSV: {os.path.join(args.out_dir, 'summary_metrics.csv')}")
    print(f"Per-image CSV: {os.path.join(args.out_dir, 'per_image_metrics.csv')}")
    print(f"Visualizations: {os.path.join(args.out_dir, '*.png')}")
    print("="*60)

if __name__ == "__main__":
    main()