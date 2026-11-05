"""
MedSAM Model Adapter
Reuses bowang-lab/MedSAM prompt-based segmentation implementation.
Supports user bounding boxes or derived bounding boxes from prior detection/segmentation.
"""

import os
import sys
import io
import time
import base64
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import torch
from PIL import Image

from src.platform.base_model import BaseMedicalModel


class MedSAMAdapter(BaseMedicalModel):
    """
    Adapter for MedSAM (Universal Medical Image Segmentation with Prompting).
    Reuses MedSAM ViT-B model from bowang-lab/MedSAM.
    """

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(config, device)
        self.model = None
        self.processor = None
        self.image_size = 1024
        self.load_model()

    def load_model(self) -> bool:
        weights_path = self.config.get("weights_path", "checkpoints/medsam/medsam_vit_b.pth")
        try:
            from transformers import SamModel, SamProcessor
            print("[MedSAMAdapter] Loading MedSAM ViT-B model...")
            self.model = SamModel.from_pretrained("wanglab/medsam-vit-base")
            self.processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("[MedSAMAdapter] MedSAM ViT-B model loaded successfully.")
            return True
        except Exception as e:
            print(f"[MedSAMAdapter] Failed to load checkpoint: {e}")
            self.is_loaded = False
            return False

    def preprocess(self, raw_input: Any) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extracts normalized 1024x1024 array and original dimensions.
        """
        if isinstance(raw_input, np.ndarray):
            pil_img = Image.fromarray(raw_input.astype(np.uint8)).convert("RGB")
        elif isinstance(raw_input, Image.Image):
            pil_img = raw_input.convert("RGB")
        elif isinstance(raw_input, str) and os.path.exists(raw_input):
            pil_img = Image.open(raw_input).convert("RGB")
        else:
            raise TypeError("Unsupported input type for MedSAM preprocessing.")

        w, h = pil_img.size
        resized = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        norm = np.array(resized, dtype=np.float32)
        return norm, (w, h)

    def infer(self, tensor_input: Any) -> Dict[str, Any]:
        """
        Executes prompt-guided inference if weights are loaded.
        """
        t0 = time.time()
        norm_img, (orig_w, orig_h) = tensor_input
        pil_img = Image.fromarray(norm_img.astype(np.uint8)).resize((orig_w, orig_h))

        # Box prompt from context or default center box
        box_prompt = (getattr(self, "_active_context", {}) or {}).get("box_prompt")
        if not box_prompt:
            box_prompt = [int(orig_w * 0.2), int(orig_h * 0.2), int(orig_w * 0.8), int(orig_h * 0.8)]

        if not self.is_loaded or self.model is None or self.processor is None:
            elapsed = round(time.time() - t0, 4)
            return {
                "mask": None,
                "orig_shape": (orig_w, orig_h),
                "box_prompt": box_prompt,
                "status": "checkpoint_required",
                "inference_duration_seconds": elapsed,
                "execution_hardware": f"CPU ({os.cpu_count()}-core)" if self.device.type == "cpu" else str(self.device)
            }

        inputs = self.processor(pil_img, input_boxes=[[box_prompt]], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        mask = masks[0][0][0].numpy().astype(np.uint8)

        elapsed = round(time.time() - t0, 4)
        return {
            "mask": mask,
            "orig_shape": (orig_w, orig_h),
            "box_prompt": box_prompt,
            "pil_img": pil_img,
            "status": "executed",
            "inference_duration_seconds": elapsed,
            "execution_hardware": f"CPU ({os.cpu_count()}-core)" if self.device.type == "cpu" else str(self.device)
        }

    def postprocess(self, raw_output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates promptable segmentation summary, mask overlay, and lesion metrics.
        """
        mask = raw_output.get("mask")
        orig_shape = raw_output["orig_shape"]
        box_prompt = raw_output.get("box_prompt")
        elapsed = raw_output["inference_duration_seconds"]
        is_ready = self.is_loaded and mask is not None

        if not is_ready:
            return {
                "model": "MedSAM (ViT-B Universal Segmentor)",
                "pipeline_status": "checkpoint_required",
                "status_badge": "Checkpoint required",
                "checkpoint_status": "MedSAM checkpoint required (~375 MB)",
                "image_resolution": list(orig_shape),
                "box_prompt_received": box_prompt,
                "mask_overlay_b64": None,
                "lesion_detected": False,
                "lesion_area_percentage": 0.0,
                "upstream_attribution": {
                    "author": "Bo Wang Lab (University of Toronto / Vector Institute)",
                    "paper": "Segment Anything in Medical Images (Nature Communications 2024)",
                    "source_url": "https://github.com/bowang-lab/MedSAM",
                    "license": "Apache 2.0"
                }
            }

        pos_pixels = int(np.sum(mask))
        total_pixels = int(orig_shape[0] * orig_shape[1])
        area_pct = round((pos_pixels / (total_pixels + 1e-8)) * 100.0, 2)

        # Generate raw binary mask base64
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        buf_m = io.BytesIO()
        mask_pil.save(buf_m, format="PNG")
        raw_mask_b64 = base64.b64encode(buf_m.getvalue()).decode("utf-8")

        # Generate overlay base64
        orig_img = raw_output.get("pil_img") or context.get("pil_image")
        mask_b64 = None
        if orig_img is not None:
            orig_np = np.array(orig_img.convert("RGB"))
            overlay = orig_np.copy()
            pos_mask = mask > 0
            overlay[pos_mask, 0] = np.clip(overlay[pos_mask, 0] * 0.4 + 34, 0, 255).astype(np.uint8)
            overlay[pos_mask, 1] = np.clip(overlay[pos_mask, 1] * 0.4 + 197, 0, 255).astype(np.uint8)
            overlay[pos_mask, 2] = np.clip(overlay[pos_mask, 2] * 0.4 + 94, 0, 255).astype(np.uint8)
            pil_ov = Image.fromarray(overlay)
            buf = io.BytesIO()
            pil_ov.save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "model": "MedSAM (ViT-B Universal Segmentor)",
            "pipeline_status": "executed",
            "is_trained_ai_segmentation": True,
            "status_badge": "Complete",
            "inference_duration_seconds": elapsed,
            "execution_hardware": raw_output["execution_hardware"],
            "image_resolution": list(orig_shape),
            "box_prompt_received": box_prompt,
            "lesion_detected": bool(pos_pixels > 0),
            "lesion_pixel_count": pos_pixels,
            "lesion_area_percentage": area_pct,
            "raw_mask_b64": raw_mask_b64,
            "mask_overlay_b64": mask_b64,
            "upstream_attribution": {
                "author": "Bo Wang Lab (University of Toronto / Vector Institute)",
                "paper": "Segment Anything in Medical Images (Nature Communications 2024)",
                "source_url": "https://github.com/bowang-lab/MedSAM",
                "license": "Apache 2.0"
            }
        }
