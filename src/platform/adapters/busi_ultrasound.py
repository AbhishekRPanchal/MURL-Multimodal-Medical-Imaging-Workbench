"""
BUSI Ultrasound Model Adapter
Adapts existing verified Breast Ultrasound models (ResNet18 classifier and MONAI UNet)
to the BaseMedicalModel unified platform interface.
"""

import os
import io
import base64
from typing import Dict, Any, Optional
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import cv2

from src.platform.base_model import BaseMedicalModel
from src.models.classification import UltrasoundTransferClassifier, UltrasoundBaselineCNN
from src.models.segmentation import UltrasoundUNet

CLASS_NAMES = ["benign", "malignant", "normal"]


class BUSIClassifierAdapter(BaseMedicalModel):
    """
    Adapter for BUSI Ultrasound ResNet-18 Classifier.
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(config, device)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = None

    def load_model(self) -> bool:
        weights_path = self.config.get("weights_path", "checkpoints/classification/best_transfer.pth")
        self.model = UltrasoundTransferClassifier(architecture="resnet18", num_classes=3, pretrained=False)
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[BUSIClassifierAdapter] Loaded weights from {weights_path}")
        else:
            print(f"[BUSIClassifierAdapter] Warning: Checkpoint {weights_path} not found. Running initialized weights.")
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        return True

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        if isinstance(raw_input, np.ndarray):
            pil_img = Image.fromarray(raw_input).convert("RGB")
        elif isinstance(raw_input, Image.Image):
            pil_img = raw_input.convert("RGB")
        else:
            raise TypeError("Unsupported input type for BUSI classifier.")

        resized = pil_img.resize((256, 256), Image.BILINEAR)
        tensor = TF.to_tensor(resized)
        tensor = self.normalize(tensor).unsqueeze(0).to(self.device)
        return tensor

    def infer(self, tensor_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1)
        return probs

    def postprocess(self, raw_output: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        probs = raw_output.squeeze().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))
            }
        }


class BUSISegmentorAdapter(BaseMedicalModel):
    """
    Adapter for BUSI Ultrasound MONAI UNet Segmentor.
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(config, device)
        self.model = None

    def load_model(self) -> bool:
        weights_path = self.config.get("weights_path", "checkpoints/segmentation/experiments/exp3_standard_dice_focal_best.pth")
        if not os.path.exists(weights_path):
            if os.path.exists("checkpoints/segmentation/best_unet.pth"):
                weights_path = "checkpoints/segmentation/best_unet.pth"
            elif os.path.exists("checkpoints/segmentation/experiments/exp3_standard_dice_focal_best.pth"):
                weights_path = "checkpoints/segmentation/experiments/exp3_standard_dice_focal_best.pth"

        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"]
            # Inspect layer keys to distinguish residual vs standard UNet
            if "config" in ckpt and "num_res_units" in ckpt["config"]:
                num_res = ckpt["config"]["num_res_units"]
            else:
                is_residual = any("residual" in k for k in state_dict.keys())
                num_res = 1 if is_residual else 0

            self.model = UltrasoundUNet(
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                num_res_units=num_res
            )
            self.model.load_state_dict(state_dict)
            print(f"[BUSISegmentorAdapter] Loaded weights from {weights_path} (num_res_units={num_res})")
        else:
            self.model = UltrasoundUNet(in_channels=1, out_channels=1)
            print(f"[BUSISegmentorAdapter] Warning: Checkpoint {weights_path} not found. Running initialized weights.")

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        return True

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        if isinstance(raw_input, np.ndarray):
            pil_img = Image.fromarray(raw_input).convert("L")
        elif isinstance(raw_input, Image.Image):
            pil_img = raw_input.convert("L")
        else:
            raise TypeError("Unsupported input type for BUSI segmentor.")

        resized = pil_img.resize((256, 256), Image.BILINEAR)
        tensor = TF.to_tensor(resized).unsqueeze(0).to(self.device)
        return tensor

    def infer(self, tensor_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.sigmoid(logits)
        return probs

    def postprocess(self, raw_output: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        threshold = context.get("threshold", 0.5)
        orig_shape = context.get("original_shape", (256, 256))
        prob_map_256 = raw_output.squeeze().cpu().numpy()
        mask_bin_256 = (prob_map_256 > threshold).astype(np.uint8)

        # Scale mask back to original resolution
        orig_w, orig_h = orig_shape
        mask_pil = Image.fromarray(mask_bin_256 * 255).resize((orig_w, orig_h), Image.NEAREST)
        mask_orig = (np.array(mask_pil) > 127).astype(np.uint8)

        lesion_pixels = int(np.sum(mask_orig))
        total_pixels = orig_w * orig_h
        area_pct = round((lesion_pixels / total_pixels) * 100, 2)

        # Derived bounding boxes
        boxes = []
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_orig)
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= 20:
                boxes.append({
                    "box_xyxy": [x, y, x + w, y + h],
                    "box_xywh": [x, y, w, h],
                    "area_pixels": area
                })

        # Generate base64 raw binary mask
        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        raw_mask_b64 = base64.b64encode(buf_mask.getvalue()).decode("utf-8")

        # Generate base64 overlay if image provided in context
        overlay_b64 = None
        orig_img = context.get("pil_image")
        if orig_img is not None:
            orig_rgb = np.array(orig_img.convert("RGB").resize((orig_w, orig_h)))
            ov = orig_rgb.copy()
            ov[mask_orig == 1, 0] = np.clip(ov[mask_orig == 1, 0] * 0.4 + 230, 0, 255).astype(np.uint8)
            ov[mask_orig == 1, 1] = np.clip(ov[mask_orig == 1, 1] * 0.4 + 30, 0, 255).astype(np.uint8)
            ov[mask_orig == 1, 2] = np.clip(ov[mask_orig == 1, 2] * 0.4 + 30, 0, 255).astype(np.uint8)
            buf_ov = io.BytesIO()
            Image.fromarray(ov).save(buf_ov, format="PNG")
            overlay_b64 = base64.b64encode(buf_ov.getvalue()).decode("utf-8")

        return {
            "lesion_detected": bool(len(boxes) > 0),
            "lesion_area_percentage": area_pct,
            "lesion_pixel_count": lesion_pixels,
            "derived_bounding_boxes": boxes,
            "mask_shape": [orig_h, orig_w],
            "raw_mask_b64": raw_mask_b64,
            "mask_overlay_b64": overlay_b64
        }
