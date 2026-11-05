"""
Controlled Segmentation Improvement Experiments
Compares:
1. Residual U-Net vs Lightweight Standard U-Net (num_res_units=0 vs 1)
2. Loss Functions: DiceCELoss vs DiceFocalLoss
Evaluates strictly on the VALIDATION fold (118 cases) to prevent test-set leakage.
"""

import os
import sys
import time
import json
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.losses import DiceCELoss, DiceFocalLoss

from src.utils.seed import set_seed, get_device
from src.dataset import BUSISegmentationDataset, ultrasound_collate_fn
from src.models.segmentation import UltrasoundUNet
from src.evaluation.metrics import compute_dice_coefficient, compute_iou

def run_experiment(
    exp_name: str,
    num_res_units: int = 1,
    loss_type: str = "dice_focal",
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 7e-4,
    seed: int = 42,
    output_dir: str = "checkpoints/segmentation/experiments"
):
    set_seed(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" EXPERIMENT: {exp_name}")
    print(f" Architecture: MONAI UNet (res_units={num_res_units})")
    print(f" Loss: {loss_type} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"=======================================================")

    train_ds = BUSISegmentationDataset(split="train", augment=True)
    val_ds = BUSISegmentationDataset(split="val", augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=ultrasound_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ultrasound_collate_fn
    )

    # Model
    model = UltrasoundUNet(
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=num_res_units,
        dropout=0.15 if num_res_units == 0 else 0.2
    ).to(device)

    # Loss
    if loss_type == "dice_ce":
        loss_fn = DiceCELoss(sigmoid=True)
    elif loss_type == "dice_focal":
        loss_fn = DiceFocalLoss(sigmoid=True, gamma=2.0)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_dice = -1.0
    best_ckpt_path = os.path.join(output_dir, f"{exp_name}_best.pth")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_benign_dice": [],
        "val_malignant_dice": [],
        "val_normal_dice": []
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_train_loss / len(train_ds)

        # Validation on 118 cases
        model.eval()
        running_val_loss = 0.0
        val_dices = []
        val_ious = []
        cls_dices = {"benign": [], "malignant": [], "normal": []}

        with torch.no_grad():
            for imgs, masks, cases in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = loss_fn(logits, masks)
                running_val_loss += loss.item() * imgs.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                for b in range(imgs.size(0)):
                    p_np = preds[b, 0].cpu().numpy()
                    m_np = masks[b, 0].cpu().numpy()
                    d = compute_dice_coefficient(p_np, m_np)
                    iou = compute_iou(p_np, m_np)
                    val_dices.append(d)
                    val_ious.append(iou)
                    c_name = cases[b]["class"]
                    cls_dices[c_name].append(d)

        val_loss = running_val_loss / len(val_ds)
        val_dice = float(np.mean(val_dices))
        val_iou = float(np.mean(val_ious))
        benign_d = float(np.mean(cls_dices["benign"])) if cls_dices["benign"] else 0.0
        malignant_d = float(np.mean(cls_dices["malignant"])) if cls_dices["malignant"] else 0.0
        normal_d = float(np.mean(cls_dices["normal"])) if cls_dices["normal"] else 0.0

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_dice"].append(round(val_dice, 4))
        history["val_iou"].append(round(val_iou, 4))
        history["val_benign_dice"].append(round(benign_d, 4))
        history["val_malignant_dice"].append(round(malignant_d, 4))
        history["val_normal_dice"].append(round(normal_d, 4))

        print(
            f"Ep [{ep:02d}/{epochs:02d}] "
            f"TrLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} (Benign: {benign_d:.4f}, Malig: {malignant_d:.4f}, Norm: {normal_d:.4f}) | "
            f"Val IoU: {val_iou:.4f}",
            flush=True
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_benign_dice": benign_d,
                "val_malignant_dice": malignant_d,
                "val_normal_dice": normal_d,
                "config": {
                    "num_res_units": num_res_units,
                    "loss_type": loss_type,
                    "lr": lr,
                    "batch_size": batch_size
                }
            }, best_ckpt_path)

    elapsed = time.time() - t0
    print(f"[{exp_name}] Completed in {elapsed:.1f}s | Best Val Dice: {best_val_dice:.4f}")

    with open(os.path.join(output_dir, f"{exp_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return best_ckpt_path, best_val_dice, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--num_res_units", type=int, default=1)
    parser.add_argument("--loss_type", type=str, default="dice_focal")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=7e-4)
    args = parser.parse_args()

    run_experiment(
        exp_name=args.exp_name,
        num_res_units=args.num_res_units,
        loss_type=args.loss_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
