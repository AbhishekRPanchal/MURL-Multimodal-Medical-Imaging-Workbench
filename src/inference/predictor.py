"""
Unified Ultrasound Inference Predictor
Loads trained classification and segmentation models for end-to-end clinical demonstration.
Produces classification probabilities, segmentation mask, lesion burden area, and bounding boxes.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.models.classification import UltrasoundTransferClassifier, UltrasoundBaselineCNN
from src.models.segmentation import UltrasoundUNet
from src.utils.seed import get_device

CLASS_NAMES = ["benign", "malignant", "normal"]

class UltrasoundPredictor:
    """
    Production-ready inference pipeline for multi-task ultrasound analysis.
    Supports CPU and GPU fallback with automated batch dimension handling.
    """
    def __init__(
        self,
        classifier_path: Optional[str] = "checkpoints/classification/best_transfer.pth",
        segmentor_path: Optional[str] = "checkpoints/segmentation/best_unet.pth",
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Load Classifier
        if classifier_path and os.path.exists(classifier_path):
            ckpt = torch.load(classifier_path, map_location=self.device)
            mtype = ckpt.get("model_type", "transfer")
            if mtype == "baseline":
                self.classifier = UltrasoundBaselineCNN(in_channels=3, num_classes=3)
            else:
                self.classifier = UltrasoundTransferClassifier(architecture="resnet18", num_classes=3, pretrained=False)
            self.classifier.load_state_dict(ckpt["model_state_dict"])
            self.classifier = self.classifier.to(self.device)
            self.classifier.eval()
            print(f"[Predictor] Loaded Classifier checkpoint: {classifier_path}")
        else:
            print(f"[Predictor] Notice: Classifier checkpoint not found at {classifier_path}. Initializing default architecture for CPU inference.")
            self.classifier = UltrasoundTransferClassifier(architecture="resnet18", num_classes=3, pretrained=False).to(self.device)
            self.classifier.eval()

        # Load Segmentor
        if segmentor_path and os.path.exists(segmentor_path):
            ckpt = torch.load(segmentor_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"]
            num_res = 1
            if "config" in ckpt and "num_res_units" in ckpt["config"]:
                num_res = ckpt["config"]["num_res_units"]
            else:
                is_residual = any("residual" in k for k in state_dict.keys())
                num_res = 1 if is_residual else 0
            self.segmentor = UltrasoundUNet(in_channels=1, out_channels=1, num_res_units=num_res)
            self.segmentor.load_state_dict(state_dict)
            self.segmentor = self.segmentor.to(self.device)
            self.segmentor.eval()
            print(f"[Predictor] Loaded Segmentor checkpoint: {segmentor_path} (num_res_units={num_res})")
        else:
            print(f"[Predictor] Notice: Segmentor checkpoint not found at {segmentor_path}. Initializing default MONAI UNet for CPU inference.")
            self.segmentor = UltrasoundUNet(in_channels=1, out_channels=1).to(self.device)
            self.segmentor.eval()

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        seg_threshold: float = 0.5
    ) -> Dict:
        """
        Runs joint classification and lesion segmentation on an input ultrasound image.
        """
        # Load and validate image
        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                orig_pil = Image.fromarray(image_input).convert("RGB")
            else:
                orig_pil = Image.fromarray(image_input.astype(np.uint8))
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
        else:
            raise TypeError("Unsupported image input type.")

        orig_w, orig_h = orig_pil.size

        result = {
            "original_resolution": [orig_w, orig_h],
            "classification": None,
            "segmentation": None
        }

        # 1. Classification
        if self.classifier is not None:
            img_cls = orig_pil.resize((256, 256), Image.BILINEAR)
            t_cls = TF.to_tensor(img_cls)
            t_cls = self.normalize(t_cls).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.classifier(t_cls)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

            result["classification"] = {
                "predicted_class": CLASS_NAMES[pred_idx],
                "confidence": round(confidence, 4),
                "probabilities": {
                    CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
                }
            }

        # 2. Segmentation
        if self.segmentor is not None:
            gray_pil = orig_pil.convert("L").resize((256, 256), Image.BILINEAR)
            t_seg = TF.to_tensor(gray_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.segmentor(t_seg)
                prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
                mask_bin_256 = (prob_map > seg_threshold).astype(np.uint8)

            # Resize mask back to original resolution
            mask_pil = Image.fromarray(mask_bin_256 * 255).resize((orig_w, orig_h), Image.NEAREST)
            mask_orig = (np.array(mask_pil) > 127).astype(np.uint8)

            # Calculate lesion statistics
            lesion_pixels = int(np.sum(mask_orig))
            total_pixels = orig_w * orig_h
            area_percentage = round((lesion_pixels / total_pixels) * 100, 2)

            # Extract derived bounding boxes from connected components
            boxes = []
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_orig)
            for i in range(1, num_labels):
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area >= 20:  # noise filter
                    boxes.append({
                        "box_xyxy": [x, y, x + w, y + h],
                        "box_xywh": [x, y, w, h],
                        "area_pixels": area
                    })

            result["segmentation"] = {
                "lesion_detected": bool(len(boxes) > 0),
                "lesion_area_percentage": area_percentage,
                "lesion_pixel_count": lesion_pixels,
                "derived_bounding_boxes": boxes,
                "mask_shape": [orig_h, orig_w]
            }

        return result
