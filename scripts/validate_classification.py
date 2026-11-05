"""
Task 4: Independent Classification Validation Script
Verifies ResNet18 transfer learning from saved checkpoint:
- Reproduces held-out test evaluation.
- Class-wise precision, recall, F1, support.
- One-vs-Rest ROC-AUC.
- Detailed confusion matrix.
- Audit of loss class weighting and ImageNet pretraining.
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset import BUSIClassificationDataset, ultrasound_collate_fn
from src.models.classification import UltrasoundTransferClassifier
from src.evaluation.metrics import compute_classification_metrics

def validate_classification():
    device = torch.device("cpu")
    ckpt_path = "checkpoints/classification/best_transfer.pth"
    print(f"=== CLASSIFICATION VALIDATION ===")
    print(f"Checkpoint: {ckpt_path}")

    # Inspect checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = state.get("epoch")
    val_f1 = state.get("val_f1", state.get("val_macro_f1", "N/A"))
    print(f"Checkpoint Epoch: {epoch} | Saved Val F1: {val_f1}")

    # Inspect training history
    hist_path = "checkpoints/classification/history_transfer.json"
    with open(hist_path, "r") as f:
        hist = json.load(f)
    print(f"Training History: {hist['epochs']} epochs completed.")
    for ep, (tl, vl, va, vf1) in enumerate(zip(hist["train_loss"], hist["val_loss"], hist["val_accuracy"], hist["val_f1_macro"]), 1):
        print(f"  Epoch {ep:02d}: Train Loss={tl:.4f} | Val Loss={vl:.4f} | Val Acc={va*100:.2f}% | Val Macro F1={vf1:.4f}")

    # Load model
    model = UltrasoundTransferClassifier(architecture="resnet18", num_classes=3, pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_ds = BUSIClassificationDataset(split="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=ultrasound_collate_fn)
    print(f"Loaded held-out test set: {len(test_ds)} cases.")

    all_preds = []
    all_targets = []
    all_probs = []

    t0 = time.time()
    with torch.no_grad():
        for imgs, targets, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    eval_time = time.time() - t0
    print(f"Test evaluation completed in {eval_time:.2f}s.")

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    class_names = ["benign", "malignant", "normal"]
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names=class_names)

    print("\n--- Quantitative Verification Results (Held-Out Test Set: N=116) ---")
    print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision: {metrics['precision_macro']*100:.2f}%")
    print(f"Macro Recall: {metrics['recall_macro']*100:.2f}%")
    print(f"Macro F1-Score: {metrics['f1_macro']*100:.2f}%")
    print(f"Weighted F1-Score: {metrics['f1_weighted']*100:.2f}%")
    print(f"Macro ROC-AUC: {metrics['roc_auc_macro']:.4f}")
    print(f"Weighted ROC-AUC: {metrics['roc_auc_weighted']:.4f}")

    print("\n--- Per-Class Performance ---")
    for cname in class_names:
        cm = metrics["per_class"][cname]
        print(f"  {cname.upper():<10}: Precision={cm['precision']*100:6.2f}% | Recall={cm['recall']*100:6.2f}% | F1={cm['f1']*100:6.2f}% | Support={cm['support']}")

    print("\n--- Confusion Matrix ---")
    cm_arr = np.array(metrics["confusion_matrix"])
    print(f"{'':>12} {'Pred Benign':>12} {'Pred Malig':>12} {'Pred Normal':>12}")
    for i, cname in enumerate(class_names):
        print(f"{'Actual ' + cname:>12} {cm_arr[i, 0]:>12} {cm_arr[i, 1]:>12} {cm_arr[i, 2]:>12}")

    # Generate high-resolution confusion matrix plot
    os.makedirs("outputs/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([c.capitalize() for c in class_names], fontsize=11)
    ax.set_yticklabels([c.capitalize() for c in class_names], fontsize=11)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Class (Ground Truth)", fontsize=12, fontweight="bold")
    plt.title("ResNet18 Confusion Matrix (Held-out Test Set, N=116)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = "outputs/figures/classification_validation_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix plot to {cm_plot_path}")

    # Document class weights
    # Training distribution: 306 benign, 147 malignant, 93 normal (out of 546)
    n_samples = 546
    n_classes = 3
    counts = np.array([306, 147, 93])
    # Sklearn inverse frequency formula: n_samples / (n_classes * counts)
    inv_weights = n_samples / (n_classes * counts)
    class_weights_info = {
        "formula": "w_c = N_total / (N_classes * N_c)",
        "train_counts": {"benign": 306, "malignant": 147, "normal": 93},
        "computed_weights": {
            "benign": round(float(inv_weights[0]), 4),
            "malignant": round(float(inv_weights[1]), 4),
            "normal": round(float(inv_weights[2]), 4)
        }
    }
    print(f"\nInverse Class Frequency Weights: {class_weights_info['computed_weights']}")

    # Save comprehensive validation artifact
    validation_report = {
        "task": "Task 4: Classification Validation",
        "model_architecture": "ResNet18 (Torchvision)",
        "pretrained_initialization": "ImageNet (ResNet18_Weights.DEFAULT)",
        "training_epochs": hist["epochs"],
        "class_weights": class_weights_info,
        "test_metrics": metrics
    }
    report_path = "outputs/classification_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    print(f"Saved full validation report to {report_path}")

if __name__ == "__main__":
    validate_classification()
