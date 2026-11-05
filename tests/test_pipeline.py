"""
Unit Tests for Reproducibility, Seeding, and Metric Calculation Edge Cases
"""

import numpy as np
import torch
import pytest
from src.utils.seed import set_seed, get_device
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_dice_coefficient,
    compute_iou
)

def test_reproducibility():
    set_seed(42)
    t1 = torch.randn(5, 5)
    set_seed(42)
    t2 = torch.randn(5, 5)
    assert torch.equal(t1, t2), "Tensors must be strictly equal with identical seed"

def test_device_fallback():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "cpu"]

def test_dice_and_iou_empty_masks():
    # Both empty masks: perfect agreement (Dice = 1.0, IoU = 1.0)
    m1 = np.zeros((10, 10))
    m2 = np.zeros((10, 10))
    assert compute_dice_coefficient(m1, m2) == 1.0
    assert compute_iou(m1, m2) == 1.0

def test_dice_and_iou_disjoint_masks():
    # Disjoint masks: zero overlap
    m1 = np.zeros((10, 10))
    m1[0:5, 0:5] = 1.0
    m2 = np.zeros((10, 10))
    m2[5:10, 5:10] = 1.0
    assert compute_dice_coefficient(m1, m2) == 0.0
    assert compute_iou(m1, m2) == 0.0

def test_classification_metrics_computation():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.3, 0.5],
        [0.2, 0.5, 0.3]
    ])
    res = compute_classification_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in res
    assert "f1_macro" in res
    assert res["accuracy"] == pytest.approx(4.0 / 6.0)
