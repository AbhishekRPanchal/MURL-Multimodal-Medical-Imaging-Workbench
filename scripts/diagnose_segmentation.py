import os, sys, json, torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import BUSISegmentationDataset, ultrasound_collate_fn
from src.models.segmentation import UltrasoundUNet
from src.evaluation.metrics import compute_dice_coefficient, compute_iou

def analyze_segmentation_failures():
    device = torch.device("cpu")
    ckpt_path = "checkpoints/segmentation/best_unet.pth"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = UltrasoundUNet(in_channels=1, out_channels=1, channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=1)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    test_ds = BUSISegmentationDataset(split="test", augment=False)
    
    print(f"Total test cases: {len(test_ds)}")

    logits_stats = []
    probs_stats = []
    dice_per_case = []
    case_diagnostics = []

    os.makedirs("outputs/failure_analysis", exist_ok=True)

    with torch.no_grad():
        for i in range(len(test_ds)):
            img, mask, case = test_ds[i]
            img_in = img.unsqueeze(0)
            logits = model(img_in)
            probs = torch.sigmoid(logits)

            logit_min = logits.min().item()
            logit_max = logits.max().item()
            logit_mean = logits.mean().item()
            prob_min = probs.min().item()
            prob_max = probs.max().item()
            prob_mean = probs.mean().item()

            pred_bin_50 = (probs > 0.5).float().squeeze().numpy()
            pred_bin_30 = (probs > 0.3).float().squeeze().numpy()
            pred_bin_20 = (probs > 0.2).float().squeeze().numpy()
            mask_np = mask.squeeze().numpy()

            d50 = compute_dice_coefficient(pred_bin_50, mask_np)
            d30 = compute_dice_coefficient(pred_bin_30, mask_np)
            d20 = compute_dice_coefficient(pred_bin_20, mask_np)
            iou50 = compute_iou(pred_bin_50, mask_np)

            pred_fg_pixels = np.sum(pred_bin_50)
            true_fg_pixels = np.sum(mask_np)

            diag = {
                "idx": i,
                "case_id": case["case_id"],
                "class": case["class"],
                "true_fg_pixels": int(true_fg_pixels),
                "pred_fg_pixels": int(pred_fg_pixels),
                "prob_min": round(prob_min, 4),
                "prob_max": round(prob_max, 4),
                "prob_mean": round(prob_mean, 4),
                "dice_th50": round(d50, 4),
                "dice_th30": round(d30, 4),
                "dice_th20": round(d20, 4),
                "iou_th50": round(iou50, 4)
            }
            case_diagnostics.append(diag)
            dice_per_case.append(d50)

    # Print summary
    print(f"Mean Test Dice (th=0.5): {np.mean(dice_per_case):.4f}")
    
    # Analyze by class
    for cname in ["benign", "malignant", "normal"]:
        c_diags = [d for d in case_diagnostics if d["class"] == cname]
        d_mean = np.mean([d["dice_th50"] for d in c_diags])
        d30_mean = np.mean([d["dice_th30"] for d in c_diags])
        pred_fg_avg = np.mean([d["pred_fg_pixels"] for d in c_diags])
        true_fg_avg = np.mean([d["true_fg_pixels"] for d in c_diags])
        p_max_avg = np.mean([d["prob_max"] for d in c_diags])
        print(f"Class {cname} (N={len(c_diags)}): Mean Dice (0.5)={d_mean:.4f} | Mean Dice (0.3)={d30_mean:.4f} | Avg True FG={true_fg_avg:.0f} | Avg Pred FG={pred_fg_avg:.0f} | Avg Max Prob={p_max_avg:.4f}")

    with open("outputs/failure_analysis/diagnostic_summary.json", "w") as f:
        json.dump(case_diagnostics, f, indent=2)

    # Generate 20 visual comparisons: Original | Ground Truth | Predicted (th=0.5)
    # Pick 10 benign, 7 malignant, 3 normal
    selected_indices = []
    benign_indices = [i for i, d in enumerate(case_diagnostics) if d["class"] == "benign"][:10]
    malignant_indices = [i for i, d in enumerate(case_diagnostics) if d["class"] == "malignant"][:7]
    normal_indices = [i for i, d in enumerate(case_diagnostics) if d["class"] == "normal"][:3]
    selected_indices = benign_indices + malignant_indices + normal_indices

    print(f"Generating 20 visual comparison panels...")
    fig, axes = plt.subplots(20, 3, figsize=(12, 60))
    for row, idx in enumerate(selected_indices):
        img, mask, case = test_ds[idx]
        with torch.no_grad():
            prob = torch.sigmoid(model(img.unsqueeze(0))).squeeze().numpy()
        pred_mask = (prob > 0.5).astype(np.float32)

        orig_img = img.squeeze().numpy()
        gt_mask = mask.squeeze().numpy()
        d_val = compute_dice_coefficient(pred_mask, gt_mask)
        iou_val = compute_iou(pred_mask, gt_mask)

        axes[row, 0].imshow(orig_img, cmap="gray")
        axes[row, 0].set_title(f"Case: {case['case_id']} ({case['class']})", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_mask, cmap="gray")
        axes[row, 1].set_title(f"Ground Truth (FG: {int(np.sum(gt_mask))} px)", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_mask, cmap="gray")
        axes[row, 2].set_title(f"Pred (Dice: {d_val:.3f}, IoU: {iou_val:.3f})", fontsize=10)
        axes[row, 2].axis("off")

    plt.tight_layout()
    comparison_20_path = "outputs/figures/segmentation_failure_20_comparisons.png"
    plt.savefig(comparison_20_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved 20-panel visual failure comparison to {comparison_20_path}")

if __name__ == "__main__":
    analyze_segmentation_failures()
