"""
Ultrasound Classification Training Script
Implements training and validation loops with class-weighted loss,
checkpointing, early stopping, and metric logging.
"""

import os
import sys
import time
import json
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.seed import set_seed, get_device
from src.dataset import BUSIClassificationDataset, ultrasound_collate_fn
from src.models.classification import UltrasoundBaselineCNN, UltrasoundTransferClassifier
from src.evaluation.metrics import compute_classification_metrics

def train_classifier(
    model_type: str = "transfer",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    seed: int = 42,
    output_dir: str = "checkpoints/classification"
):
    set_seed(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n==================================================")
    print(f" TRAINING CLASSIFIER: {model_type.upper()}")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"==================================================")

    # Prepare datasets & loaders
    train_dataset = BUSIClassificationDataset(split="train", augment=True)
    val_dataset = BUSIClassificationDataset(split="val", augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=ultrasound_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=ultrasound_collate_fn
    )

    # Class imbalance handling
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"[Dataset] Train class counts: {train_dataset.get_class_counts()}")
    print(f"[Dataset] Computed Inverse Class Weights: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize model
    if model_type == "baseline":
        model = UltrasoundBaselineCNN(in_channels=3, num_classes=3, dropout_rate=0.3)
    elif model_type == "transfer":
        model = UltrasoundTransferClassifier(architecture="resnet18", num_classes=3, pretrained=True, dropout_rate=0.3)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_checkpoint_path = os.path.join(output_dir, f"best_{model_type}.pth")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": [],
        "epochs": epochs,
        "model_type": model_type
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                running_val_loss += loss.item() * images.size(0)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(labels.cpu().numpy())

        scheduler.step()

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_probs_arr = np.concatenate(all_probs, axis=0)
        val_preds_arr = np.concatenate(all_preds, axis=0)
        val_targets_arr = np.concatenate(all_targets, axis=0)

        metrics = compute_classification_metrics(val_targets_arr, val_preds_arr, val_probs_arr)
        val_acc = metrics["accuracy"]
        val_f1 = metrics["f1_macro"]

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_f1_macro"].append(round(val_f1, 4))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val F1 (Macro): {val_f1:.4f}",
            flush=True
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "class_weights": class_weights.cpu()
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Val F1: {val_f1:.4f})", flush=True)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Val F1: {best_val_f1:.4f}")

    # Save training history
    history_path = os.path.join(output_dir, f"history_{model_type}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_checkpoint_path, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="transfer", choices=["baseline", "transfer"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_classifier(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
