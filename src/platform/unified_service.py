"""
Unified Multi-Modal Inference Service
High-level orchestrator integrating Format Validation, DICOM De-Identification,
NIfTI Volumetric Parsing, Modality Detection, Model Compatibility Checking, and Routed Inference.
"""

import io
import os
import base64
from typing import Dict, Any, Optional, Union
from PIL import Image
import numpy as np
import torch

from src.platform.format_validator import ImageFormatValidator
from src.platform.dicom_parser import DICOMMetadataParser
from src.platform.volume_parser import NIfTIMetadataParser
from src.platform.modality_detector import ModalityDetector
from src.platform.registry import ModelRegistry
from src.platform.router import ModelRouter
from src.platform.base_model import ModelPredictionOutput
from src.utils.seed import get_device


class UnifiedInferenceService:
    """
    Central service orchestrating multi-modal medical scan ingestion and AI inference.
    Supports PNG, JPG, DICOM, and NIfTI 3D/4D formats.
    """

    def __init__(self, registry_path: str = "model_registry/registry.yaml", device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.registry = ModelRegistry(registry_path=registry_path)
        self.router = ModelRouter(registry=self.registry, device=self.device)

    def process_scan(
        self,
        file_bytes: bytes,
        filename: str,
        user_modality_override: Optional[str] = None,
        requested_model_id: Optional[str] = None,
        requested_task: Optional[str] = None,
        anatomical_region: Optional[str] = None,
        seg_threshold: float = 0.5,
        box_prompt: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end ingestion and inference on an uploaded medical file.

        Args:
            file_bytes: Raw binary file bytes.
            filename: File name including extension.
            user_modality_override: Optional user confirmation/override of modality.
            requested_model_id: Optional model_id to route specifically.
            requested_task: 'classification', 'segmentation', 'planning', etc.
            seg_threshold: Segmentation binarization threshold.
            box_prompt: Optional [x0, y0, x1, y1] bounding box prompt for promptable models (MedSAM).

        Returns:
            Dict containing validation, modality metadata, model results, and provenance.
        """
        # Step 1: Format Validation
        format_report = ImageFormatValidator.validate(file_bytes, filename)
        if not format_report["is_valid"]:
            return {
                "status": "error",
                "stage": "format_validation",
                "error": format_report["error"],
                "format_report": format_report
            }

        resolved_format = format_report["format"]
        is_volumetric = format_report.get("is_volumetric", False)

        # Step 2: Modality Detection & Metadata Parsing
        dicom_metadata = None
        nifti_metadata = None
        pil_image = None
        raw_volume_array = None
        nifti_temp_path = None

        if resolved_format == "dicom":
            dicom_metadata = DICOMMetadataParser.parse_metadata(file_bytes)
            modality_info = ModalityDetector.detect_from_dicom(file_bytes)
            pil_image = DICOMMetadataParser.extract_displayable_image(file_bytes)
            raw_volume_array = DICOMMetadataParser.extract_pixel_array(file_bytes)

        elif resolved_format in ["nifti", "nifti_gz"]:
            nifti_parsed = NIfTIMetadataParser.parse_from_bytes(file_bytes, filename)
            modality_info = ModalityDetector.detect(file_bytes, filename)
            pil_image = nifti_parsed["preview_slices"]["axial"]
            
            # Serialize preview slices to base64 for frontend display
            def pil_to_base64(img: Image.Image) -> str:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")

            nifti_metadata = {
                "format": nifti_parsed["format"],
                "filename": nifti_parsed["filename"],
                "dimensions": nifti_parsed["dimensions"],
                "spatial_ndim": nifti_parsed["spatial_ndim"],
                "voxel_spacing_mm": nifti_parsed["voxel_spacing_mm"],
                "anatomical_orientation": nifti_parsed["anatomical_orientation"],
                "data_type": nifti_parsed["data_type"],
                "intensity_distribution": nifti_parsed["intensity_distribution"],
                "orthogonal_previews": {
                    "axial_b64": pil_to_base64(nifti_parsed["preview_slices"]["axial"]),
                    "coronal_b64": pil_to_base64(nifti_parsed["preview_slices"]["coronal"]),
                    "sagittal_b64": pil_to_base64(nifti_parsed["preview_slices"]["sagittal"])
                }
            }
            # For 3D volumetric adapters, persist temporary file for direct NIfTI consumption
            import tempfile, nibabel as nib
            suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(file_bytes)
            tmp.close()
            nifti_temp_path = tmp.name
            raw_volume_array = nifti_temp_path
        else:
            # Standard raster image (PNG, JPG)
            pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            modality_info = ModalityDetector.detect(file_bytes, filename)

        # Automatic modality detection is authoritative by default.
        detected_mod = modality_info.get("modality", "Unknown")
        analysis_mode = "Automatic"
        detected_anatomy = anatomical_region
        if not detected_anatomy and dicom_metadata:
            detected_anatomy = dicom_metadata.get("body_part_examined")

        if user_modality_override and user_modality_override.strip():
            override_val = user_modality_override.strip()
            analysis_mode = f"Manual — {override_val}"
            mapped_mod = override_val

            if "ultrasound" in override_val.lower():
                mapped_mod = "Ultrasound"
                if "thyroid" in override_val.lower():
                    detected_anatomy = "Thyroid"
                elif "breast" in override_val.lower():
                    detected_anatomy = "Breast"
            elif "x-ray" in override_val.lower() or "xray" in override_val.lower():
                mapped_mod = "X-ray"
                detected_anatomy = "Chest"
            elif "ct" in override_val.lower():
                mapped_mod = "CT"
            elif "mri" in override_val.lower():
                mapped_mod = "MRI"
                if "brain" in override_val.lower():
                    detected_anatomy = "Brain"

            # Prevent dangerous silent cross-modality routing (e.g. X-ray -> Ultrasound)
            is_both_us = ("ultrasound" in detected_mod.lower() and "ultrasound" in mapped_mod.lower())
            is_mr_ct_ambiguity = ("mri" in detected_mod.lower() or "ct" in detected_mod.lower()) and ("mri" in mapped_mod.lower() or "ct" in mapped_mod.lower())

            if detected_mod not in ["Unknown", "Other", "Unknown / Volumetric"] and detected_mod.lower() != mapped_mod.lower() and not is_both_us and not is_mr_ct_ambiguity:
                raise ValueError(
                    f"Modality conflict: Automatic detection identified scan as '{detected_mod}', but manual override requested '{override_val}'. "
                    f"Incompatible cross-modality routing is rejected for clinical safety."
                )

            active_modality = mapped_mod
        else:
            active_modality = detected_mod

        if requested_model_id == "medsam_vit_b" or box_prompt is not None:
            analysis_mode = "Region segmentation"

        if requested_model_id and not requested_model_id.strip():
            requested_model_id = None

        # Step 3: Scan Properties for Compatibility Checking
        w, h = pil_image.size if pil_image else (0, 0)

        # If user explicitly requested a model, align speculative modalities/anatomy
        if requested_model_id and not user_modality_override:
            try:
                target_meta = self.router.registry.get_model(requested_model_id)
                model_mod = target_meta.get("modality")
                if model_mod and model_mod != "Multi-Modal":
                    if not modality_info.get("is_segmentation_mask") and active_modality != "Segmentation Mask" and (modality_info.get("needs_user_confirmation") or "mri/ct" in str(active_modality).lower()):
                        active_modality = model_mod
                if not detected_anatomy:
                    detected_anatomy = target_meta.get("anatomical_region")
            except Exception:
                pass

        scan_properties = {
            "modality": active_modality,
            "anatomical_region": detected_anatomy,
            "format": resolved_format,
            "spatial_ndim": 3 if is_volumetric else 2,
            "num_channels": 3 if pil_image and pil_image.mode == "RGB" else 1,
            "mri_sequences": []
        }

        # Step 4: Routing and Execution
        try:
            adapter, compat_report = self.router.route(
                scan_properties=scan_properties,
                requested_model_id=requested_model_id,
                requested_task=requested_task
            )

            # Context for inference
            context = {
                "threshold": seg_threshold,
                "original_shape": (w, h),
                "box_prompt": box_prompt,
                "pil_image": pil_image,
                "affine": np.array(nifti_metadata.get("affine", np.eye(4))) if nifti_metadata else np.eye(4)
            }

            # Select model input representation (2D Image or 3D Volume)
            if scan_properties["spatial_ndim"] == 3 and raw_volume_array is not None:
                model_input = raw_volume_array
            else:
                model_input = pil_image

            prediction_output = adapter.predict(model_input, context=context)

            return {
                "status": "success",
                "format_report": format_report,
                "modality_detection": modality_info,
                "active_modality": active_modality,
                "compatibility_report": compat_report,
                "dicom_metadata": dicom_metadata,
                "nifti_metadata": nifti_metadata,
                "analysis_mode": analysis_mode,
                "prediction": prediction_output.model_dump(),
                "regulatory_notice": "Research use only. Not for clinical diagnostic decision-making."
            }

        except Exception as e:
            return {
                "status": "error",
                "stage": "routing_or_inference",
                "error": str(e),
                "format_report": format_report,
                "modality_detection": modality_info,
                "active_modality": active_modality,
                "analysis_mode": analysis_mode,
                "dicom_metadata": dicom_metadata,
                "nifti_metadata": nifti_metadata,
                "regulatory_notice": "Research use only. Not for clinical diagnostic decision-making."
            }
        finally:
            if nifti_temp_path and os.path.exists(nifti_temp_path):
                try:
                    os.remove(nifti_temp_path)
                except Exception:
                    pass
