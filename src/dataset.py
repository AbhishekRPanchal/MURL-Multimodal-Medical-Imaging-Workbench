"""
Dataset Loading, Mask Fusion, and Ultrasound Preprocessing Pipeline
Provides high-performance PyTorch Dataset classes for both classification and segmentation
with in-memory caching and ultrasound-specific geometric/radiometric augmentations.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CLASS_TO_IDX = {"benign": 0, "malignant": 1, "normal": 2}
IDX_TO_CLASS = {0: "benign", 1: "malignant", 2: "normal"}

def load_case_mask(mask_paths: List[str], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Loads one or more mask files for a given case and fuses them via logical OR.
    Normal cases with no lesions or empty masks return all-zero arrays.
    Returns: binary uint8 numpy array of shape (H, W) with values 0 and 1.
    """
    if not mask_paths:
        raise ValueError("No mask paths provided")

    fused_mask = None
    for mp in mask_paths:
        with Image.open(mp) as m_img:
            m_img = m_img.convert("L")
            if target_size is not None:
                m_img = m_img.resize(target_size, Image.NEAREST)
            arr = np.array(m_img, dtype=np.uint8)
            bin_mask = (arr > 127).astype(np.uint8)
            if fused_mask is None:
                fused_mask = bin_mask
            else:
                fused_mask = np.bitwise_or(fused_mask, bin_mask)

    return fused_mask

def ultrasound_collate_fn(batch):
    """Custom collate function to handle variable length metadata safely."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch]
    return images, targets, metas

class BUSIClassificationDataset(Dataset):
    """
    Ultrasound classification dataset with in-memory caching.
    """
    def __init__(
        self,
        metadata_path: str = os.path.join("data", "processed", "dataset_metadata.json"),
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == "train")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Run src/data_audit.py first.")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        all_cases = meta["cases"]
        if split == "all":
            self.cases = all_cases
        else:
            self.cases = [c for c in all_cases if c.get("split") == split]

        if len(self.cases) == 0:
            raise ValueError(f"No cases found for split: {split}")

        # In-memory image cache
        self.cache = {}
        for idx, case in enumerate(self.cases):
            p = case.get("path") or case.get("image_path")
            with Image.open(p) as img:
                self.cache[idx] = img.convert("RGB").resize(self.image_size, Image.BILINEAR)

        # Standard ImageNet normalization for transfer learning models
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.cases)

    def get_class_counts(self) -> Dict[str, int]:
        counts = {"benign": 0, "malignant": 0, "normal": 0}
        for c in self.cases:
            counts[c["class"]] += 1
        return counts

    def get_class_weights(self) -> torch.FloatTensor:
        counts = self.get_class_counts()
        total = sum(counts.values())
        weights = [total / (len(counts) * counts[IDX_TO_CLASS[i]]) for i in range(len(counts))]
        return torch.tensor(weights, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        case = self.cases[idx]
        img = self.cache[idx].copy()

        # Ultrasound-appropriate augmentations applied on the 256x256 image
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
            angle = float(torch.empty(1).uniform_(-10, 10).item())
            img = TF.rotate(img, angle)
            if torch.rand(1).item() > 0.5:
                brightness_factor = float(torch.empty(1).uniform_(0.85, 1.15).item())
                contrast_factor = float(torch.empty(1).uniform_(0.85, 1.15).item())
                img = TF.adjust_brightness(img, brightness_factor)
                img = TF.adjust_contrast(img, contrast_factor)

        img_tensor = TF.to_tensor(img)
        img_tensor = self.normalize(img_tensor)
        label = CLASS_TO_IDX[case["class"]]
        return img_tensor, torch.tensor(label, dtype=torch.long), case

class BUSISegmentationDataset(Dataset):
    """
    Ultrasound lesion segmentation dataset with in-memory caching.
    """
    def __init__(
        self,
        metadata_path: str = os.path.join("data", "processed", "dataset_metadata.json"),
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True
    ):
        super().__init__()
        self.image_size = image_size
        self.split = split
        self.augment = augment and (split == "train")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Run src/data_audit.py first.")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        all_cases = meta["cases"]
        if split == "all":
            self.cases = all_cases
        else:
            self.cases = [c for c in all_cases if c.get("split") == split]

        if len(self.cases) == 0:
            raise ValueError(f"No cases found for split: {split}")

        # In-memory cache for images and fused masks
        self.img_cache = {}
        self.mask_cache = {}
        for idx, case in enumerate(self.cases):
            p = case.get("path") or case.get("image_path")
            with Image.open(p) as img:
                self.img_cache[idx] = img.convert("L").resize(self.image_size, Image.BILINEAR)

            fused_mask = load_case_mask(case["mask_paths"], target_size=self.image_size)
            self.mask_cache[idx] = Image.fromarray(fused_mask * 255)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        case = self.cases[idx]
        img = self.img_cache[idx].copy()
        mask_pil = self.mask_cache[idx].copy()

        # Synchronized augmentation
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
                mask_pil = TF.hflip(mask_pil)
            angle = float(torch.empty(1).uniform_(-10, 10).item())
            img = TF.rotate(img, angle)
            mask_pil = TF.rotate(mask_pil, angle)
            if torch.rand(1).item() > 0.5:
                brightness_factor = float(torch.empty(1).uniform_(0.85, 1.15).item())
                img = TF.adjust_brightness(img, brightness_factor)

        img_tensor = TF.to_tensor(img)  # shape [1, H, W] in [0, 1]
        mask_tensor = TF.to_tensor(mask_pil)  # shape [1, H, W] in [0, 1]
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor, case
