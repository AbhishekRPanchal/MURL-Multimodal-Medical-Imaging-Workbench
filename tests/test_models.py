"""
Unit Tests for Classification and MONAI Segmentation Models
"""

import torch
import pytest
from src.models.classification import UltrasoundBaselineCNN, UltrasoundTransferClassifier
from src.models.segmentation import UltrasoundUNet

def test_baseline_cnn_forward():
    model = UltrasoundBaselineCNN(in_channels=3, num_classes=3)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"

def test_transfer_classifier_forward():
    model = UltrasoundTransferClassifier(architecture="resnet18", num_classes=3, pretrained=False)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"

def test_monai_unet_forward():
    model = UltrasoundUNet(in_channels=1, out_channels=1)
    dummy_input = torch.randn(2, 1, 256, 256)
    out = model(dummy_input)
    assert out.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {out.shape}"
