"""
MONAI BraTS Multi-Parametric Brain MRI Tumor Segmentation Adapter
Reuses Project-MONAI SegResNet architecture for brain glioma sub-region segmentation.
Evaluates multi-sequence MRI: T1, T1ce, T2, FLAIR.
Quantifies Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
"""

import os
import io
import time
import base64
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import torch
from PIL import Image
import nibabel as nib

from src.platform.base_model import BaseMedicalModel

try:
    import monai
    from monai.networks.nets import SegResNet
    from monai.inferers import sliding_window_inference
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


class MONAIBraTSAdapter(BaseMedicalModel):
    """
    Adapter for multi-parametric Brain MRI Tumor Segmentation (BraTS).
    Segments glioma sub-compartments using MONAI SegResNet.
    """

    SUB_REGIONS = {
        0: "Tumor Core (TC)",
        1: "Whole Tumor (WT)",
        2: "Enhancing Tumor (ET)"
    }

    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(config, device)
        self.model = None
        self.spatial_dims = 3
        self.in_channels = 4
        self.out_channels = 3
        self.roi_size = (64, 64, 64)
        self.has_checkpoint = False
        self.load_model()

    def load_model(self) -> bool:
        if not HAS_MONAI:
            print("[MONAIBraTSAdapter] MONAI package not available.")
            return False

        try:
            self.model = SegResNet(
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                init_filters=16,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                dropout_prob=0.2
            )

            weights_path = self.config.get("weights_path", "checkpoints/mri/brats_mri_segresnet.pt")
            if not os.path.exists(weights_path) and os.path.exists("checkpoints/mri/brats_mri_segresnet.pt"):
                weights_path = "checkpoints/mri/brats_mri_segresnet.pt"

            if weights_path and os.path.exists(weights_path):
                print(f"[MONAIBraTSAdapter] Loading genuine SegResNet weights from {weights_path}")
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get("state_dict", checkpoint)
                self.model.load_state_dict(state_dict)
                self.has_checkpoint = True
                print("[MONAIBraTSAdapter] Genuine BraTS SegResNet checkpoint loaded successfully.")
            else:
                self.has_checkpoint = False
                print(f"[MONAIBraTSAdapter] Checkpoint not found at {weights_path}.")

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[MONAIBraTSAdapter] Load exception: {e}")
            self.is_loaded = False
            return False

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """
        Accepts:
        - 4D numpy array (4, H, W, D) or (H, W, D, 4)
        - 3D numpy array (H, W, D) -> replicated across 4 sequences
        - File path to NIfTI volume
        Standardizes intensities using non-zero z-score normalization per channel.
        """
        if isinstance(raw_input, str) and os.path.exists(raw_input):
            import nibabel as nib
            nii = nib.load(raw_input)
            arr = np.asanyarray(nii.dataobj).astype(np.float32)
        elif isinstance(raw_input, np.ndarray):
            arr = raw_input.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input for MONAIBraTSAdapter: {type(raw_input)}")

        # Ensure (4, H, W, D) format
        if arr.ndim == 3:
            # Single sequence provided: replicate across all 4 sequences (T1, T1ce, T2, FLAIR)
            channels = [arr.copy() for _ in range(4)]
            vol_4d = np.stack(channels, axis=0)
        elif arr.ndim == 4:
            if arr.shape[0] == 4:
                vol_4d = arr
            elif arr.shape[-1] == 4:
                vol_4d = np.transpose(arr, (3, 0, 1, 2))
            else:
                raise ValueError(f"Expected 4 channels for BraTS MRI, found shape: {arr.shape}")
        else:
            raise ValueError(f"Invalid dimensions for MRI volume: {arr.ndim}")

        # BraTS non-zero intensity normalization per channel
        normalized_channels = []
        for c in range(4):
            ch = vol_4d[c]
            mask = ch > 0
            if np.any(mask):
                mean = float(np.mean(ch[mask]))
                std = float(np.std(ch[mask])) + 1e-6
                ch_norm = np.zeros_like(ch)
                ch_norm[mask] = (ch[mask] - mean) / std
            else:
                ch_norm = ch
            normalized_channels.append(ch_norm)

        normalized_vol = np.stack(normalized_channels, axis=0)

        # Shape into (1, 4, H, W, D)
        tensor = torch.from_numpy(normalized_vol).unsqueeze(0).float()
        return tensor.to(self.device)

    def infer(self, tensor_input: torch.Tensor) -> Dict[str, Any]:
        """
        Runs neural forward pass on 4-channel MRI volume using loaded SegResNet weights.
        """
        t0 = time.time()
        if self.model is None or not self.is_loaded:
            raise RuntimeError("MONAIBraTSAdapter model is not loaded.")

        _, _, h, w, d = tensor_input.shape

        if not self.has_checkpoint:
            elapsed = round(time.time() - t0, 4)
            vol_preview = tensor_input[0, 0].cpu().numpy()
            return {
                "probs": None,
                "vol_preview": vol_preview,
                "input_shape": list(tensor_input.shape),
                "inference_duration_seconds": elapsed,
                "execution_hardware": f"CPU ({os.cpu_count()}-core)" if self.device.type == "cpu" else str(self.device)
            }

        with torch.no_grad():
            if min(h, w, d) < 32:
                out = self.model(tensor_input)
            elif h > 64 or w > 64 or d > 64:
                out = sliding_window_inference(
                    inputs=tensor_input,
                    roi_size=self.roi_size,
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=0.25
                )
            else:
                out = self.model(tensor_input)

            probs = torch.sigmoid(out).squeeze(0).cpu().numpy()

        elapsed = round(time.time() - t0, 4)
        vol_preview = tensor_input[0, 0].cpu().numpy()
        return {
            "probs": probs,
            "vol_preview": vol_preview,
            "input_shape": list(tensor_input.shape),
            "inference_duration_seconds": elapsed,
            "execution_hardware": f"CPU ({os.cpu_count()}-core)" if self.device.type == "cpu" else str(self.device)
        }

    def postprocess(self, raw_output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thresholds probabilities into binary masks and computes volumetric compartment metrics.
        """
        probs = raw_output.get("probs")
        elapsed = raw_output["inference_duration_seconds"]
        vol_preview = raw_output.get("vol_preview")
        _, _, h, w, d = raw_output["input_shape"]

        def render_gray_b64(slc: np.ndarray) -> str:
            p1, p99 = np.percentile(slc, 1), np.percentile(slc, 99)
            denom = p99 - p1 if p99 > p1 else 1.0
            norm = np.clip((slc - p1) / denom * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(norm.T)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        slice_stacks = {"axial": [], "coronal": [], "sagittal": [], "total_counts": {"axial": 0, "coronal": 0, "sagittal": 0}}
        if vol_preview is not None:
            slice_stacks = {
                "axial": [render_gray_b64(vol_preview[:, :, z]) for z in range(d)],
                "coronal": [render_gray_b64(vol_preview[:, y, :]) for y in range(w)],
                "sagittal": [render_gray_b64(vol_preview[x, :, :]) for x in range(h)],
                "total_counts": {"axial": d, "coronal": w, "sagittal": h}
            }

        if probs is None or not self.has_checkpoint:
            return {
                "model": "MONAI SegResNet (BraTS Brain MRI Multi-Sequence)",
                "pipeline_status": "checkpoint_required",
                "status_badge": "Checkpoint required",
                "checkpoint_status": "MRI model checkpoint required",
                "checkpoint_message": "Pretrained BraTS checkpoint not installed.",
                "execution_hardware": raw_output["execution_hardware"],
                "inference_duration_seconds": elapsed,
                "tumor_detected": False,
                "compartments": [],
                "primary_volumes_cm3": {
                    "whole_tumor": 0.0,
                    "tumor_core": 0.0,
                    "enhancing_tumor": 0.0
                },
                "multiplanar_previews": {},
                "slice_stacks": slice_stacks,
                "required_sequences": ["T1", "T1ce", "T2", "FLAIR"],
                "clinical_disclaimer": "RESEARCH USE ONLY: Pretrained BraTS SegResNet weights required for automated tumor segmentation.",
                "upstream_attribution": {
                    "author": "Project-MONAI Consortium & BraTS Challenge",
                    "paper": "The Multimodal Brain Tumor Image Segmentation Benchmark (IEEE TMI)",
                    "source_url": "https://huggingface.co/MONAI/brats_mri_segmentation",
                    "license": "Apache 2.0"
                }
            }

        # Genuine inference postprocessing
        tc_mask = (probs[0] > 0.5).astype(np.uint8)
        wt_mask = (probs[1] > 0.5).astype(np.uint8)
        et_mask = (probs[2] > 0.5).astype(np.uint8)

        # Enforce nested topological consistency: ET is subset of TC, TC is subset of WT
        tc_mask = np.maximum(tc_mask, et_mask)
        wt_mask = np.maximum(wt_mask, tc_mask)

        voxel_vol_cm3 = 0.001  # 1mm3 = 0.001 cm3
        wt_voxels = int(np.sum(wt_mask))
        tc_voxels = int(np.sum(tc_mask))
        et_voxels = int(np.sum(et_mask))
        edema_voxels = int(np.sum((wt_mask == 1) & (tc_mask == 0)))
        ncr_voxels = int(np.sum((tc_mask == 1) & (et_mask == 0)))

        # Standard BraTS multi-label volume: 1=NCR/NET, 2=ED, 3=ET
        multilabel_vol = np.zeros_like(wt_mask, dtype=np.uint8)
        multilabel_vol[(tc_mask == 1) & (et_mask == 0)] = 1
        multilabel_vol[(wt_mask == 1) & (tc_mask == 0)] = 2
        multilabel_vol[et_mask == 1] = 3

        # Persist multi-label NIfTI mask with preserved geometry
        out_nifti_path = None
        try:
            mri_out_dir = os.path.abspath("outputs/mri_segmentations")
            os.makedirs(mri_out_dir, exist_ok=True)
            out_nifti_path = os.path.join(mri_out_dir, f"brats_mri_seg_{int(time.time())}.nii.gz")
            input_affine = context.get("affine", np.eye(4)) if context else np.eye(4)
            nib_mri = nib.Nifti1Image(multilabel_vol, input_affine)
            nib.save(nib_mri, out_nifti_path)
        except Exception as e:
            print(f"[MONAIBraTSAdapter] Note: could not persist NIfTI mask: {e}")

        compartments = [
            {
                "compartment": "Whole Tumor (WT)",
                "description": "Entire pathological abnormal region including edema, active core, and necrosis",
                "voxel_count": wt_voxels,
                "volume_cm3": round(wt_voxels * voxel_vol_cm3, 3)
            },
            {
                "compartment": "Tumor Core (TC)",
                "description": "Gross tumor volume including enhancing tumor and necrotic/cystic components",
                "voxel_count": tc_voxels,
                "volume_cm3": round(tc_voxels * voxel_vol_cm3, 3)
            },
            {
                "compartment": "Enhancing Tumor (ET)",
                "description": "Hypervascularized active rim of blood-brain barrier disruption",
                "voxel_count": et_voxels,
                "volume_cm3": round(et_voxels * voxel_vol_cm3, 3)
            },
            {
                "compartment": "Peritumoral Edema (ED)",
                "description": "Surrounding vasogenic fluid accumulation without solid tumor infiltration",
                "voxel_count": edema_voxels,
                "volume_cm3": round(edema_voxels * voxel_vol_cm3, 3)
            },
            {
                "compartment": "Necrotic & Non-Enhancing Core (NCR/NET)",
                "description": "Hypovascular devitalized core within tumor center",
                "voxel_count": ncr_voxels,
                "volume_cm3": round(ncr_voxels * voxel_vol_cm3, 3)
            }
        ]

        # Multi-planar slice visualization with tumor overlay
        mid_axial_prob = probs[1, :, :, d // 2]
        mid_coronal_prob = probs[1, :, w // 2, :]
        mid_sagittal_prob = probs[1, h // 2, :, :]

        def render_tumor_overlay_b64(slc: np.ndarray, prob_map: np.ndarray) -> str:
            p1, p99 = np.percentile(slc, 1), np.percentile(slc, 99)
            denom = p99 - p1 if p99 > p1 else 1.0
            norm = np.clip((slc - p1) / denom * 255.0, 0, 255).astype(np.uint8)
            rgb = np.stack([norm, norm, norm], axis=-1)
            # Overlay red/orange on tumor regions (prob > 0.5)
            tumor_mask = prob_map > 0.5
            if np.any(tumor_mask):
                rgb[tumor_mask, 0] = np.clip(rgb[tumor_mask, 0] * 0.4 + 220, 0, 255).astype(np.uint8)
                rgb[tumor_mask, 1] = np.clip(rgb[tumor_mask, 1] * 0.4 + 38, 0, 255).astype(np.uint8)
                rgb[tumor_mask, 2] = np.clip(rgb[tumor_mask, 2] * 0.4 + 38, 0, 255).astype(np.uint8)
            pil = Image.fromarray(np.transpose(rgb, (1, 0, 2)))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        axial_ov = render_tumor_overlay_b64(vol_preview[:, :, d // 2], mid_axial_prob)
        coronal_ov = render_tumor_overlay_b64(vol_preview[:, w // 2, :], mid_coronal_prob)
        sagittal_ov = render_tumor_overlay_b64(vol_preview[h // 2, :, :], mid_sagittal_prob)

        return {
            "model": "MONAI SegResNet (BraTS Multi-Parametric Brain MRI Segmentor)",
            "pipeline_status": "executed",
            "is_trained_ai_segmentation": True,
            "status_badge": "Complete",
            "execution_hardware": raw_output["execution_hardware"],
            "inference_duration_seconds": elapsed,
            "tumor_detected": bool(wt_voxels > 50),
            "compartments": compartments,
            "primary_volumes_cm3": {
                "whole_tumor": round(wt_voxels * voxel_vol_cm3, 3),
                "tumor_core": round(tc_voxels * voxel_vol_cm3, 3),
                "enhancing_tumor": round(et_voxels * voxel_vol_cm3, 3)
            },
            "output_nifti_path": out_nifti_path,
            "multiplanar_previews": {
                "axial_wt_b64": axial_ov,
                "coronal_wt_b64": coronal_ov,
                "sagittal_wt_b64": sagittal_ov
            },
            "mask_overlay_b64": axial_ov,
            "slice_stacks": slice_stacks,
            "required_sequences": ["T1", "T1ce", "T2", "FLAIR"],
            "clinical_disclaimer": "RESEARCH DEMONSTRATION ONLY: Quantitative volumetric measurements are computed via SegResNet for academic research evaluation.",
            "upstream_attribution": {
                "author": "Project-MONAI Consortium & BraTS Challenge",
                "paper": "The Multimodal Brain Tumor Image Segmentation Benchmark (IEEE TMI)",
                "source_url": "https://huggingface.co/MONAI/brats_mri_segmentation",
                "license": "Apache 2.0"
            }
        }
