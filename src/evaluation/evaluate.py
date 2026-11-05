"""
Evaluation Pipeline on Held-out Test Set
Evaluates both classification and segmentation models strictly on the held-out test split.
Generates publication-quality figures: Confusion Matrix, ROC curves, Loss Curves,
and Ground Truth vs Predicted Mask Overlays.
"""

import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

from src.utils.seed import set_seed, get_device
from src.dataset import (
    BUSIClassificationDataset,
    BUSISegmentationDataset,
    ultrasound_collate_fn,
    CLASS_TO_IDX,
    IDX_TO_CLASS
)
from src.models.classification import UltrasoundBaselineCNN, UltrasoundTransferClassifier
from src.models.segmentation import UltrasoundUNet
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_dice_coefficient,
    compute_iou
)

FIGURES_DIR = os.path.join("outputs", "figures")
OUTPUTS_DIR = os.path.join("outputs")
os.makedirs(FIGURES_DIR, exist_ok=True)

def evaluate_classification_model(model_type: str, checkpoint_path: str, device: torch.device):
    """Evaluates a classification checkpoint on the held-out test set."""
    print(f"\n[Test Evaluation] Evaluating {model_type.upper()} on held-out test set...")
    
    test_dataset = BUSIClassificationDataset(split="test", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=ultrasound_collate_fn)

    # Initialize model
    if model_type == "baseline":
        model = UltrasoundBaselineCNN(in_channels=3, num_classes=3)
    elif model_type == "transfer":
        model = UltrasoundTransferClassifier(architecture="resnet18", num_classes=3, pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []
    cases = []

    with torch.no_grad():
        for images, labels, batch_cases in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())
            cases.extend(batch_cases)

    probs_arr = np.concatenate(all_probs, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    metrics = compute_classification_metrics(targets_arr, preds_arr, probs_arr)

    # Plot Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({model_type.capitalize()} on Held-Out Test Set)")
    plt.colorbar()
    tick_marks = np.arange(3)
    class_names = ["Benign", "Malignant", "Normal"]
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_type}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot ROC Curves
    plt.figure(figsize=(7, 6))
    for i, cname in enumerate(class_names):
        y_bin = (targets_arr == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs_arr[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC = {roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {model_type.capitalize()} (Held-out Test)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(FIGURES_DIR, f"roc_curve_{model_type}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    return metrics, cm_path, roc_path

def evaluate_segmentation_model(checkpoint_path: str, device: torch.device):
    """Evaluates MONAI UNet on the held-out test set and generates visual comparisons."""
    print(f"\n[Test Evaluation] Evaluating MONAI U-Net on held-out test set...")
    
    test_dataset = BUSISegmentationDataset(split="test", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=ultrasound_collate_fn)

    model = UltrasoundUNet(in_channels=1, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    dices = []
    ious = []
    per_class_dice = {"benign": [], "malignant": [], "normal": []}
    per_class_iou = {"benign": [], "malignant": [], "normal": []}

    visual_samples = []

    with torch.no_grad():
        for i, (image, mask, case) in enumerate(test_loader):
            image = image.to(device)
            mask = mask.to(device)

            logits = model(image)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()

            pred_np = pred.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            img_np = image.squeeze().cpu().numpy()

            d = compute_dice_coefficient(pred_np, mask_np)
            iou = compute_iou(pred_np, mask_np)

            dices.append(d)
            ious.append(iou)

            case_cls = case[0]["class"]
            case_id = case[0]["case_id"]
            per_class_dice[case_cls].append(d)
            per_class_iou[case_cls].append(iou)

            # Collect representative samples for visual comparison (first 2 of each class)
            if len([s for s in visual_samples if s["class"] == case_cls]) < 2:
                visual_samples.append({
                    "class": case_cls,
                    "case_id": case_id,
                    "image": img_np,
                    "mask_gt": mask_np,
                    "mask_pred": pred_np,
                    "dice": d,
                    "iou": iou
                })

    test_dice_mean = float(np.mean(dices))
    test_iou_mean = float(np.mean(ious))

    print(f"Test Overall Dice: {test_dice_mean:.4f} | IoU: {test_iou_mean:.4f}")
    for cls in ["benign", "malignant", "normal"]:
        print(f"  Class {cls}: Dice={np.mean(per_class_dice[cls]):.4f}, IoU={np.mean(per_class_iou[cls]):.4f}")

    # Generate Visual Comparison Figure
    fig, axes = plt.subplots(len(visual_samples), 4, figsize=(16, 3.5 * len(visual_samples)))
    for idx, s in enumerate(visual_samples):
        # 1. Original Ultrasound
        axes[idx, 0].imshow(s["image"], cmap="gray")
        axes[idx, 0].set_title(f"Original ({s['case_id']})")
        axes[idx, 0].axis("off")

        # 2. Ground Truth Mask
        axes[idx, 1].imshow(s["image"], cmap="gray")
        axes[idx, 1].imshow(s["mask_gt"], cmap="jet", alpha=0.45)
        axes[idx, 1].set_title(f"Ground Truth ({s['class']})")
        axes[idx, 1].axis("off")

        # 3. Predicted Mask
        axes[idx, 2].imshow(s["image"], cmap="gray")
        axes[idx, 2].imshow(s["mask_pred"], cmap="plasma", alpha=0.45)
        axes[idx, 2].set_title(f"Predicted Mask\nDice: {s['dice']:.3f} | IoU: {s['iou']:.3f}")
        axes[idx, 2].axis("off")

        # 4. Error Overlay: Green=TP, Red=FP, Blue=FN
        tp = np.logical_and(s["mask_gt"] == 1, s["mask_pred"] == 1)
        fp = np.logical_and(s["mask_gt"] == 0, s["mask_pred"] == 1)
        fn = np.logical_and(s["mask_gt"] == 1, s["mask_pred"] == 0)
        
        rgb_overlay = np.stack([s["image"]]*3, axis=-1)
        # Apply color tints
        rgb_overlay[tp] = [0.0, 1.0, 0.0]  # Green: True Positive
        rgb_overlay[fp] = [1.0, 0.0, 0.0]  # Red: False Positive
        rgb_overlay[fn] = [0.0, 0.4, 1.0]  # Blue: False Negative

        axes[idx, 3].imshow(rgb_overlay)
        axes[idx, 3].set_title("Overlap Analysis\n(Grn=TP, Red=FP, Blu=FN)")
        axes[idx, 3].axis("off")

    plt.tight_layout()
    seg_fig_path = os.path.join(FIGURES_DIR, "segmentation_test_comparisons.png")
    plt.savefig(seg_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    metrics = {
        "overall_dice": test_dice_mean,
        "overall_iou": test_iou_mean,
        "class_breakdown": {
            cls: {
                "dice": float(np.mean(per_class_dice[cls])),
                "iou": float(np.mean(per_class_iou[cls])),
                "count": len(per_class_dice[cls])
            } for cls in ["benign", "malignant", "normal"]
        }
    }
    return metrics, seg_fig_path

def plot_training_curves():
    """Generates training loss and validation metric curves for all experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Classification Baseline
    base_hist_path = os.path.join("checkpoints", "classification", "history_baseline.json")
    trans_hist_path = os.path.join("checkpoints", "classification", "history_transfer.json")
    unet_hist_path = os.path.join("checkpoints", "segmentation", "history_unet.json")

    # Plot 1: Classification Loss Comparison
    if os.path.exists(base_hist_path):
        with open(base_hist_path) as f:
            h_base = json.load(f)
            axes[0].plot(range(1, len(h_base["train_loss"]) + 1), h_base["train_loss"], label="Baseline Train Loss", linestyle="--", color="gray")
            axes[0].plot(range(1, len(h_base["val_loss"]) + 1), h_base["val_loss"], label="Baseline Val Loss", color="blue")
    if os.path.exists(trans_hist_path):
        with open(trans_hist_path) as f:
            h_trans = json.load(f)
            axes[0].plot(range(1, len(h_trans["train_loss"]) + 1), h_trans["train_loss"], label="ResNet18 Train Loss", linestyle="--", color="orange")
            axes[0].plot(range(1, len(h_trans["val_loss"]) + 1), h_trans["val_loss"], label="ResNet18 Val Loss", color="red")
    axes[0].set_title("Classification Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Classification Macro F1 Comparison
    if os.path.exists(base_hist_path) and os.path.exists(trans_hist_path):
        axes[1].plot(range(1, len(h_base["val_f1_macro"]) + 1), h_base["val_f1_macro"], marker="o", label="Baseline CNN Val F1", color="blue")
        axes[1].plot(range(1, len(h_trans["val_f1_macro"]) + 1), h_trans["val_f1_macro"], marker="s", label="ResNet18 Val F1", color="red")
    axes[1].set_title("Validation Macro F1-Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Segmentation MONAI UNet Loss & Dice
    if os.path.exists(unet_hist_path):
        with open(unet_hist_path) as f:
            h_unet = json.load(f)
            ep = range(1, len(h_unet["train_loss"]) + 1)
            ax3 = axes[2]
            ax3.plot(ep, h_unet["train_loss"], label="Train DiceCELoss", linestyle="--", color="purple")
            ax3.plot(ep, h_unet["val_loss"], label="Val DiceCELoss", color="darkviolet")
            ax3.set_ylabel("Loss")
            ax3.set_xlabel("Epoch")
            
            ax3_twin = ax3.twinx()
            ax3_twin.plot(ep, h_unet["val_dice"], label="Val Dice", color="green", marker="^")
            ax3_twin.set_ylabel("Dice Score", color="green")
            ax3.set_title("MONAI U-Net Segmentation Convergence")
            ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(FIGURES_DIR, "training_curves_comparison.png")
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    return curves_path

def run_full_evaluation():
    set_seed(42)
    device = get_device()

    report = {
        "evaluation_type": "Held-Out Test Set Evaluation",
        "total_test_cases": 116,
        "classification": {},
        "segmentation": {}
    }

    # Evaluate Baseline CNN
    base_ckpt = os.path.join("checkpoints", "classification", "best_baseline.pth")
    if os.path.exists(base_ckpt):
        metrics_base, _, _ = evaluate_classification_model("baseline", base_ckpt, device)
        report["classification"]["baseline_cnn"] = metrics_base

    # Evaluate Transfer Learning (ResNet18)
    trans_ckpt = os.path.join("checkpoints", "classification", "best_transfer.pth")
    if os.path.exists(trans_ckpt):
        metrics_trans, _, _ = evaluate_classification_model("transfer", trans_ckpt, device)
        report["classification"]["resnet18_transfer"] = metrics_trans

    # Evaluate MONAI UNet
    unet_ckpt = os.path.join("checkpoints", "segmentation", "best_unet.pth")
    if os.path.exists(unet_ckpt):
        metrics_unet, _ = evaluate_segmentation_model(unet_ckpt, device)
        report["segmentation"]["monai_unet"] = metrics_unet

    # Plot training curves
    plot_training_curves()

    report_path = os.path.join(OUTPUTS_DIR, "test_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Evaluation Complete] Full report saved to: {report_path}")
    return report

if __name__ == "__main__":
    run_full_evaluation()
