"""
Unit Tests for Ultrasound Dataset Loaders and Transforms
"""

import os
import torch
import pytest
from src.dataset import BUSIClassificationDataset, BUSISegmentationDataset, load_case_mask

METADATA_PATH = os.path.join("data", "processed", "dataset_metadata.json")

def test_metadata_exists():
    assert os.path.exists(METADATA_PATH), "dataset_metadata.json must exist. Run data audit first."

def test_classification_dataset_shapes_and_types():
    dataset = BUSIClassificationDataset(metadata_path=METADATA_PATH, split="train", augment=False)
    assert len(dataset) > 0, "Classification dataset should have samples."
    
    img, label, meta = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 256, 256), f"Expected shape (3, 256, 256), got {img.shape}"
    assert isinstance(label, torch.Tensor)
    assert label.item() in [0, 1, 2], "Label must be in [0, 1, 2]"
    assert "case_id" in meta

def test_classification_class_weights():
    dataset = BUSIClassificationDataset(metadata_path=METADATA_PATH, split="train")
    weights = dataset.get_class_weights()
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (3,)
    assert (weights > 0).all(), "Class weights must be strictly positive"

def test_segmentation_dataset_shapes_and_binarization():
    dataset = BUSISegmentationDataset(metadata_path=METADATA_PATH, split="train", augment=False)
    assert len(dataset) > 0, "Segmentation dataset should have samples."

    img, mask, meta = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 256, 256), f"Expected image shape (1, 256, 256), got {img.shape}"
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (1, 256, 256), f"Expected mask shape (1, 256, 256), got {mask.shape}"
    
    unique_vals = torch.unique(mask).tolist()
    for val in unique_vals:
        assert val in [0.0, 1.0], f"Mask must be binary (0.0 or 1.0), found {val}"
