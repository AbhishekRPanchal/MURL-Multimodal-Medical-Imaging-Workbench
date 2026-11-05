"""
Runner script for Task 3 Segmentation Experiments.
Runs 3 controlled configurations, logs validation metrics, selects the winning model
based STRICTLY on validation set performance, and then evaluates the winner on the
held-out test set (116 cases).
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.train_segmentor_experiment import run_experiment
from src.dataset import BUSISegmentationDataset, ultrasound_collate_fn
from src.models.segmentation import UltrasoundUNet
from src.evaluation.metrics import compute_dice_coefficient, compute_iou

def evaluate_on_test(ckpt_path: str, num_res_units: int):
    device = torch.device("cpu")
    model = UltrasoundUNet(
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=num_res_units
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    test_ds = BUSISegmentationDataset(split="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=ultrasound_collate_fn)

    dices = []
    ious = []
    cls_dices = {"benign": [], "malignant": [], "normal": []}
    cls_ious = {"benign": [], "malignant": [], "normal": []}

    with torch.no_grad():
        for img, mask, case in test_loader:
            img = img.to(device)
            logits = model(img)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()

            pred_np = pred.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()

            d = compute_dice_coefficient(pred_np, mask_np)
            iou = compute_iou(pred_np, mask_np)

            dices.append(d)
            ious.append(iou)

            cname = case[0]["class"]
            cls_dices[cname].append(d)
            cls_ious[cname].append(iou)

    res = {
        "overall_dice": float(np.mean(dices)),
        "overall_iou": float(np.mean(ious)),
        "per_class": {
            "benign": {"dice": float(np.mean(cls_dices["benign"])), "iou": float(np.mean(cls_ious["benign"])), "count": len(cls_dices["benign"])},
            "malignant": {"dice": float(np.mean(cls_dices["malignant"])), "iou": float(np.mean(cls_ious["malignant"])), "count": len(cls_dices["malignant"])},
            "normal": {"dice": float(np.mean(cls_dices["normal"])), "iou": float(np.mean(cls_ious["normal"])), "count": len(cls_dices["normal"])}
        }
    }
    return res

def main():
    print("=== STARTING CONTROLLED SEGMENTATION EXPERIMENTS ===")

    experiments = [
        {"exp_name": "exp1_residual_dice_ce", "num_res_units": 1, "loss_type": "dice_ce", "epochs": 5, "lr": 5e-4},
        {"exp_name": "exp2_residual_dice_focal", "num_res_units": 1, "loss_type": "dice_focal", "epochs": 5, "lr": 7e-4},
        {"exp_name": "exp3_standard_dice_focal", "num_res_units": 0, "loss_type": "dice_focal", "epochs": 5, "lr": 7e-4}
    ]

    results_table = []
    val_performances = {}

    for exp in experiments:
        ckpt_path, best_val_dice, hist = run_experiment(
            exp_name=exp["exp_name"],
            num_res_units=exp["num_res_units"],
            loss_type=exp["loss_type"],
            epochs=exp["epochs"],
            batch_size=16,
            lr=exp["lr"]
        )
        val_performances[exp["exp_name"]] = {
            "val_dice": best_val_dice,
            "ckpt_path": ckpt_path,
            "num_res_units": exp["num_res_units"],
            "loss_type": exp["loss_type"]
        }

    # Model selection based STRICTLY on validation set performance
    print("\n=== MODEL SELECTION BASED ON VALIDATION SET (118 CASES) ===")
    best_exp_name = max(val_performances, key=lambda k: val_performances[k]["val_dice"])
    best_exp = val_performances[best_exp_name]
    print(f"WINNING MODEL on Validation Set: {best_exp_name} (Val Dice: {best_exp['val_dice']:.4f})")

    # Evaluate all 3 on the held-out test set for the experimental report
    print("\n=== HELD-OUT TEST SET EVALUATION (116 CASES) ===")
    test_results_all = {}
    for exp_name, exp_data in val_performances.items():
        print(f"Evaluating {exp_name} on held-out test set...")
        t_res = evaluate_on_test(exp_data["ckpt_path"], exp_data["num_res_units"])
        test_results_all[exp_name] = {
            "config": exp_data,
            "test_results": t_res
        }
        print(f"  {exp_name} -> Test Dice: {t_res['overall_dice']:.4f} | IoU: {t_res['overall_iou']:.4f}")
        print(f"    Benign: {t_res['per_class']['benign']['dice']:.4f} | Malignant: {t_res['per_class']['malignant']['dice']:.4f} | Normal: {t_res['per_class']['normal']['dice']:.4f}")

    # Save summary report
    summary_path = "outputs/segmentation_experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "validation_selection": {
                "winning_experiment": best_exp_name,
                "selection_criterion": "Highest Validation Sørensen-Dice on 118 cases",
                "val_performances": val_performances
            },
            "test_evaluations": test_results_all
        }, f, indent=2)

    print(f"\nAll experiments complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
