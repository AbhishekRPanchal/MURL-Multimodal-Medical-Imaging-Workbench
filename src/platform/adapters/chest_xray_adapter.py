"""
Chest X-Ray Multi-Pathology Classification Adapter
Reuses torchxrayvision (MILA / Stanford) DenseNet-121 multi-dataset models.
Supports 18 thoracic pathologies (Atelectasis, Cardiomegaly, Effusion, Pneumonia, etc.).
"""

import os
import time
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
from PIL import Image

from src.platform.base_model import BaseMedicalModel

try:
    import torchxrayvision as xrv
    HAS_XRV = True
except ImportError:
    HAS_XRV = False


class ChestXRayAdapter(BaseMedicalModel):
    """
    Adapter for Chest X-ray multi-pathology classification.
    Evaluates 18 thoracic radiological findings using DenseNet-121.
    """

    DEFAULT_PATHOLOGIES = [
        "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
        "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
        "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
        "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
    ]

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(config, device)
        self.model = None
        self.pathologies = self.DEFAULT_PATHOLOGIES
        self.op_threshs = None
        self.weights_tag = self.config.get("weights_tag", "densenet121-res224-all")
        self.weights_cache = self.config.get("weights_cache", "checkpoints/xray")
        self.load_model()

    def load_model(self) -> bool:
        if not HAS_XRV:
            print("[ChestXRayAdapter] torchxrayvision package not available.")
            return False

        try:
            weights_path = self.config.get("weights_path")
            cache_dir = os.path.dirname(weights_path) if (weights_path and os.path.dirname(weights_path)) else self.weights_cache
            os.makedirs(cache_dir, exist_ok=True)

            self.model = xrv.models.DenseNet(
                weights=self.weights_tag,
                cache_dir=cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            self.pathologies = list(self.model.pathologies)
            if hasattr(self.model, "op_threshs") and self.model.op_threshs is not None:
                self.op_threshs = self.model.op_threshs.cpu().numpy().tolist()

            self.is_loaded = True
            print(f"[ChestXRayAdapter] Successfully loaded DenseNet-121 ({len(self.pathologies)} pathologies).")
            return True
        except Exception as e:
            print(f"[ChestXRayAdapter] Initialization notice: {e}")
            self.is_loaded = False
            return False

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """
        Accepts image path, PIL Image, or numpy array.
        Normalizes to 224x224 single-channel in range [-1024, 1024].
        """
        if isinstance(raw_input, str):
            img = Image.open(raw_input).convert("L")
            arr = np.array(img, dtype=np.float32)
        elif isinstance(raw_input, Image.Image):
            arr = np.array(raw_input.convert("L"), dtype=np.float32)
        elif isinstance(raw_input, np.ndarray):
            arr = raw_input.astype(np.float32)
            if arr.ndim == 3:
                # If RGB or (C, H, W)
                if arr.shape[0] in (1, 3, 4):
                    arr = np.mean(arr[:3], axis=0)
                else:
                    arr = np.mean(arr[:, :, :3], axis=2)
            elif arr.ndim > 3:
                # 3D/4D volume - take center axial slice
                arr = arr[:, :, arr.shape[2] // 2]
        else:
            raise TypeError(f"Unsupported input type for ChestXRayAdapter: {type(raw_input)}")

        # Resize to 224x224
        pil_im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8) if arr.max() <= 255 else ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8))
        pil_im = pil_im.resize((224, 224), Image.Resampling.BILINEAR)
        resized_arr = np.array(pil_im, dtype=np.float32)

        # Normalize to [-1024, 1024]
        if HAS_XRV:
            normalized = xrv.datasets.normalize(resized_arr, 255.0)
        else:
            normalized = (resized_arr / 255.0) * 2048.0 - 1024.0

        # Tensor shape (1, 1, 224, 224)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
        return tensor.to(self.device)

    def infer(self, preprocessed_input: torch.Tensor) -> Dict[str, Any]:
        """
        Runs neural forward pass on normalized radiograph.
        """
        t0 = time.time()
        if self.model is None or not self.is_loaded:
            raise RuntimeError("ChestXRayAdapter model is not loaded.")

        with torch.no_grad():
            logits = self.model(preprocessed_input)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        elapsed = round(time.time() - t0, 4)
        return {
            "probs": probs,
            "inference_duration_seconds": elapsed,
            "execution_hardware": f"CPU ({os.cpu_count()}-core)" if self.device.type == "cpu" else str(self.device)
        }

    def postprocess(self, raw_output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats probabilities into ranked findings and operating threshold comparisons.
        """
        probs = raw_output["probs"]
        elapsed = raw_output["inference_duration_seconds"]

        findings = []
        for idx, name in enumerate(self.pathologies):
            prob = float(probs[idx])
            thresh = float(self.op_threshs[idx]) if (self.op_threshs and idx < len(self.op_threshs)) else 0.5
            findings.append({
                "pathology": name,
                "probability": round(prob, 4),
                "probability_percent": round(prob * 100, 2),
                "operating_threshold": round(thresh, 4),
                "threshold_exceeded": bool(prob >= thresh),
                "relative_elevation": round(prob / (thresh + 1e-6), 2)
            })

        # Sort descending by probability
        findings.sort(key=lambda x: x["probability"], reverse=True)

        # Active radiological alerts (exceeding operating threshold)
        flagged = [f for f in findings if f["threshold_exceeded"]]

        return {
            "model": "TorchXRayVision DenseNet-121 (Multi-Dataset Pretrained)",
            "pipeline_status": "executed",
            "execution_hardware": raw_output["execution_hardware"],
            "inference_duration_seconds": elapsed,
            "pathologies_evaluated": len(findings),
            "flagged_pathologies_count": len(flagged),
            "flagged_pathologies": flagged,
            "top_findings": findings[:5],
            "all_findings": findings,
            "clinical_disclaimer": "RESEARCH & INVESTIGATIONAL USE ONLY. Not certified for primary medical diagnosis. Clinical correlation by a board-certified radiologist required.",
            "upstream_attribution": {
                "author": "Joseph Paul Cohen et al. (Stanford University & MILA)",
                "paper": "TorchXRayVision: An Open-Source Library for Chest X-ray Medical Imaging AI",
                "source_url": "https://github.com/mlmed/torchxrayvision",
                "license": "Apache 2.0"
            }
        }
