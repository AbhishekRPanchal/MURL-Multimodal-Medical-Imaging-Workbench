"""
Medical Imaging Evaluation Metrics
Implements standard quantitative evaluation criteria for medical ultrasound
classification and lesion segmentation.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

# -------------------------------------------------------------
# CLASSIFICATION METRICS
# -------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = ["benign", "malignant", "normal"]
) -> Dict[str, Union[float, Dict[str, float], np.ndarray]]:
    """
    Computes comprehensive classification metrics.
    Args:
        y_true: 1D array of ground truth class indices.
        y_pred: 1D array of predicted class indices.
        y_prob: 2D array of predicted class probabilities [N, num_classes].
        class_names: List of class names.
    """
    acc = float(accuracy_score(y_true, y_pred))
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Weighted metrics
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    # Per-class metrics
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_results = {}
    for i, name in enumerate(class_names):
        per_class_results[name] = {
            "precision": float(p_cls[i]),
            "recall": float(r_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(supp_cls[i])
        }

    # ROC-AUC (One-vs-Rest)
    try:
        roc_auc_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc_macro = float("nan")
        roc_auc_weighted = float("nan")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_wt),
        "recall_weighted": float(r_wt),
        "f1_weighted": float(f1_wt),
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist()
    }


# -------------------------------------------------------------
# SEGMENTATION METRICS
# -------------------------------------------------------------

def compute_dice_coefficient(y_pred_bin: np.ndarray, y_true_bin: np.ndarray, eps: float = 1e-7) -> float:
    """
    Computes Sørensen-Dice coefficient between binary predicted mask and ground truth mask.
    Handles empty masks correctly according to medical benchmark standards:
    - If both pred and true are empty (all 0s): Dice = 1.0 (true negative match)
    - If one is empty and other has positive pixels: Dice = 0.0
    """
    pred = (y_pred_bin > 0.5).astype(np.float32)
    true = (y_true_bin > 0.5).astype(np.float32)

    pred_sum = np.sum(pred)
    true_sum = np.sum(true)

    if pred_sum == 0 and true_sum == 0:
        return 1.0
    if (pred_sum == 0 and true_sum > 0) or (true_sum == 0 and pred_sum > 0):
        return 0.0

    intersection = np.sum(pred * true)
    return float((2.0 * intersection) / (pred_sum + true_sum + eps))

def compute_iou(y_pred_bin: np.ndarray, y_true_bin: np.ndarray, eps: float = 1e-7) -> float:
    """
    Computes Jaccard index (Intersection over Union).
    """
    pred = (y_pred_bin > 0.5).astype(np.float32)
    true = (y_true_bin > 0.5).astype(np.float32)

    pred_sum = np.sum(pred)
    true_sum = np.sum(true)

    if pred_sum == 0 and true_sum == 0:
        return 1.0
    if (pred_sum == 0 and true_sum > 0) or (true_sum == 0 and pred_sum > 0):
        return 0.0

    intersection = np.sum(pred * true)
    union = pred_sum + true_sum - intersection
    return float(intersection / (union + eps))

def compute_segmentation_batch_metrics(
    preds_probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes average Dice and IoU over a batch of predictions and targets.
    Args:
        preds_probs: torch.Tensor of shape [B, 1, H, W] in [0, 1].
        targets: torch.Tensor of shape [B, 1, H, W] in {0, 1}.
        threshold: Binarization cutoff.
    """
    preds_np = (preds_probs > threshold).detach().cpu().numpy()
    targets_np = (targets > threshold).detach().cpu().numpy()

    batch_size = preds_np.shape[0]
    dices = []
    ious = []

    for i in range(batch_size):
        dices.append(compute_dice_coefficient(preds_np[i, 0], targets_np[i, 0]))
        ious.append(compute_iou(preds_np[i, 0], targets_np[i, 0]))

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious))
    }
