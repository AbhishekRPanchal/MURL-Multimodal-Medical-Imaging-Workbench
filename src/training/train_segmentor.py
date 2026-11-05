"""
Ultrasound Lesion Segmentation Training Script
Implements MONAI U-Net training with DiceCELoss, batch metrics,
checkpointing, and training curves logging.
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
from monai.losses import DiceCELoss

from src.utils.seed import set_seed, get_device
from src.dataset import BUSISegmentationDataset, ultrasound_collate_fn
from src.models.segmentation import UltrasoundUNet
from src.evaluation.metrics import compute_segmentation_batch_metrics

def train_segmentor(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 5e-4,
    seed: int = 42,
    output_dir: str = "checkpoints/segmentation"
):
    set_seed(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f" TRAINING MONAI ULTRASOUND U-NET SEGMENTATION")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"==================================================")

    # Prepare datasets & loaders
    train_dataset = BUSISegmentationDataset(split="train", augment=True)
    val_dataset = BUSISegmentationDataset(split="val", augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=ultrasound_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=ultrasound_collate_fn
    )

    print(f"[Dataset] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # MONAI combined loss: Dice + CrossEntropy with sigmoid activation for binary lesion mask
    loss_function = DiceCELoss(sigmoid=True)

    # Initialize MONAI U-Net
    model = UltrasoundUNet(
        in_channels=1,
        out_channels=1
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = -1.0
    best_checkpoint_path = os.path.join(output_dir, "best_unet.pth")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "epochs": epochs
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_dices = []
        val_ious = []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = loss_function(logits, masks)
                running_val_loss += loss.item() * images.size(0)

                pred_probs = torch.sigmoid(logits)
                batch_metrics = compute_segmentation_batch_metrics(pred_probs, masks, threshold=0.5)
                val_dices.append(batch_metrics["dice"])
                val_ious.append(batch_metrics["iou"])

        scheduler.step()

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_dice = float(np.mean(val_dices))
        epoch_val_iou = float(np.mean(val_ious))

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_dice"].append(round(epoch_val_dice, 4))
        history["val_iou"].append(round(epoch_val_iou, 4))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Dice: {epoch_val_dice:.4f} | "
            f"Val IoU: {epoch_val_iou:.4f}",
            flush=True
        )

        # Save best model
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": epoch_val_dice,
                "val_iou": epoch_val_iou
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Val Dice: {epoch_val_dice:.4f})", flush=True)

    elapsed = time.time() - start_time
    print(f"\nSegmentation training completed in {elapsed:.1f}s. Best Val Dice: {best_val_dice:.4f}")

    # Save training history
    history_path = os.path.join(output_dir, "history_unet.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_checkpoint_path, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_segmentor(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
