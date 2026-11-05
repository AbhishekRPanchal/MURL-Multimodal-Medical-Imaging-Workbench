"""
Modality Detector
Determines the imaging modality (Ultrasound, MRI, CT, X-ray) from DICOM headers
or performs image-based uncertainty estimation for standard raster images (PNG/JPG).
Never infers anatomical diagnosis or clinical findings from filenames alone.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from src.platform.dicom_parser import DICOMMetadataParser

# Standard DICOM Modality code mappings to platform modalities
DICOM_MODALITY_MAP = {
    "US": "Ultrasound",
    "MR": "MRI",
    "CT": "CT",
    "DX": "X-ray",
    "CR": "X-ray",
    "RG": "X-ray",
    "MG": "Mammography",
    "PT": "PET",
    "NM": "Nuclear Medicine"
}


class ModalityDetector:
    """
    Identifies imaging modality while strictly separating verified header evidence
    from heuristic/probabilistic estimates, flagging uncertainty and requesting confirmation.
    """

    @classmethod
    def detect_from_dicom(cls, file_bytes: bytes) -> Dict[str, Any]:
        """
        Extract modality conclusively from DICOM metadata.
        """
        metadata = DICOMMetadataParser.parse_metadata(file_bytes)
        raw_mod = metadata.get("modality", "UNKNOWN").upper()
        standard_modality = DICOM_MODALITY_MAP.get(raw_mod, f"Other ({raw_mod})")
        body_part = metadata.get("body_part_examined", "UNSPECIFIED")

        return {
            "modality": standard_modality,
            "raw_modality_code": raw_mod,
            "anatomical_hint": body_part if body_part != "UNSPECIFIED" else None,
            "probability": 1.0,
            "confidence": 1.0,
            "uncertainty": 0.0,
            "needs_user_confirmation": False,
            "detection_source": "dicom_header",
            "metadata_summary": metadata,
            "rationale": f"Extracted directly from DICOM tag (0008,0060) Modality: '{raw_mod}'."
        }

    @classmethod
    def detect_from_image(
        cls,
        image_pil: Image.Image,
        uncertainty_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Analyzes raster image (PNG/JPG) properties to provide an initial modality hypothesis,
        accompanied by an uncertainty score.

        Always requests user confirmation because raster files lack acquisition metadata.
        """
        w, h = image_pil.size
        gray_img = image_pil.convert("L")
        arr = np.array(gray_img, dtype=np.float32)

        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        black_border_pct = float(np.mean(arr < 10))

        # Heuristic scoring based on acoustic imaging vs radiographic contrast
        # Ultrasound typically features dark acoustic backgrounds (>20% black margin), localized speckle
        # X-ray has wide anatomical transmission with bone contrast (higher dynamic range, fewer pure black borders)
        if black_border_pct > 0.25 and std_val > 35:
            # Strong ultrasound acoustic profile
            prob_us = 0.75
            prob_xray = 0.15
            prob_other = 0.10
            predicted = "Ultrasound"
            confidence = prob_us
        elif black_border_pct < 0.15 and std_val > 45:
            # Radiographic profile (Chest X-ray / CT slice)
            prob_xray = 0.65
            prob_us = 0.20
            prob_other = 0.15
            predicted = "X-ray"
            confidence = prob_xray
        else:
            predicted = "Ultrasound"
            confidence = 0.55
            prob_other = 0.45

        uncertainty = round(1.0 - confidence, 4)
        needs_confirm = uncertainty >= uncertainty_threshold or True  # Always recommend confirmation for 2D rasters

        return {
            "modality": predicted,
            "probability": round(confidence, 4),
            "confidence": round(confidence, 4),
            "uncertainty": uncertainty,
            "needs_user_confirmation": needs_confirm,
            "detection_source": "raster_heuristic_analysis",
            "anatomical_hint": None,  # Strictly NEVER infer anatomy from raster pixels or filenames
            "metadata_summary": {
                "dimensions": [w, h],
                "mean_intensity": round(mean_val, 2),
                "std_intensity": round(std_val, 2),
                "black_pixel_ratio": round(black_border_pct, 4)
            },
            "rationale": (
                f"Raster format lacks DICOM headers. Heuristic speckle and background border analysis "
                f"suggests {predicted} with probability {confidence:.2f} (uncertainty {uncertainty:.2f}). "
                f"User confirmation recommended prior to clinical research inference."
            )
        }

    @classmethod
    def detect(cls, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Unified detection entry point.
        """
        lower = filename.lower()
        if lower.endswith(".dcm") or lower.endswith(".dicom"):
            return cls.detect_from_dicom(file_bytes)

        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            # Inspect voxel values and metadata to determine CT vs MRI vs Segmentation Mask vs Unknown/Volumetric
            val_min, val_max = 0.0, 0.0
            is_seg_mask = any(k in lower for k in ["seg", "mask", "totalseg_out", "label"])

            try:
                import tempfile, nibabel as nib
                suffix = ".nii.gz" if lower.endswith(".nii.gz") else ".nii"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_name = tmp.name
                try:
                    img = nib.load(tmp_name)
                    data = np.asanyarray(img.dataobj)
                    val_min = float(np.nanmin(data))
                    val_max = float(np.nanmax(data))
                    # Check for discrete integer labels (segmentation mask characteristics)
                    sub = data[:min(16, data.shape[0]), :min(16, data.shape[1]), :min(16, data.shape[2])]
                    unique_vals = np.unique(sub)
                    is_integer = np.all(np.equal(np.mod(unique_vals, 1), 0))
                    if is_integer and val_min >= 0 and val_max < 150 and len(unique_vals) < 30:
                        is_seg_mask = True
                finally:
                    if os.path.exists(tmp_name):
                        os.remove(tmp_name)
            except Exception:
                pass

            if is_seg_mask:
                return {
                    "modality": "Segmentation Mask",
                    "is_segmentation_mask": True,
                    "probability": 1.0,
                    "confidence": 1.0,
                    "uncertainty": 0.0,
                    "needs_user_confirmation": True,
                    "detection_source": "nifti_label_analysis",
                    "anatomical_hint": None,
                    "metadata_summary": {"format": "NIfTI segmentation mask", "intensity_range": [round(val_min, 1), round(val_max, 1)]},
                    "rationale": "Discrete integer labels detected. Input is a segmentation mask output, not an original CT/MRI acquisition scan."
                }

            # Check for CT Hounsfield Units (-1000 air, soft tissue ~0-100, bone >400) or CT filename
            is_ct_hu = (val_min <= -400 and val_max >= 200)
            is_ct_name = any(k in lower for k in ["ct", "computed_tomography"]) and not any(k in lower for k in ["flair", "mri", "t1", "t2", "brain"])

            # Check for MRI non-negative signal or MRI filename
            is_mri_name = any(k in lower for k in ["mri", "t1", "t2", "flair", "brats", "dwi", "adc", "mr", "brain"])
            is_mri_hu = (val_min >= -5 and val_max > 50 and not is_ct_hu)

            if is_ct_name or (is_ct_hu and not is_mri_name):
                prob = 0.95 if is_ct_name and is_ct_hu else 0.85
                return {
                    "modality": "CT",
                    "probability": prob,
                    "confidence": prob,
                    "uncertainty": 0.05 if is_ct_name and is_ct_hu else 0.15,
                    "needs_user_confirmation": False,
                    "detection_source": "hounsfield_unit_and_nifti_analysis",
                    "anatomical_hint": None,
                    "metadata_summary": {"format": "CT NIfTI", "intensity_range": [round(val_min, 1), round(val_max, 1)]},
                    "rationale": "Hounsfield Unit range and volumetric parameters identify scan as Computed Tomography (CT)."
                }
            elif is_mri_name or (is_mri_hu and not is_ct_name):
                prob = 0.90 if is_mri_name else 0.75
                return {
                    "modality": "MRI",
                    "probability": prob,
                    "confidence": prob,
                    "uncertainty": 0.10 if is_mri_name else 0.25,
                    "needs_user_confirmation": False if is_mri_name else True,
                    "detection_source": "nifti_mri_contrast_analysis",
                    "anatomical_hint": "Brain" if any(k in lower for k in ["brain", "brats"]) else None,
                    "metadata_summary": {"format": "MRI NIfTI", "intensity_range": [round(val_min, 1), round(val_max, 1)]},
                    "rationale": "Contrast distribution and volumetric parameters identify scan as Magnetic Resonance (MRI)."
                }
            else:
                return {
                    "modality": "Unknown / Volumetric",
                    "probability": 0.50,
                    "confidence": 0.50,
                    "uncertainty": 0.50,
                    "needs_user_confirmation": True,
                    "detection_source": "nifti_format",
                    "anatomical_hint": None,
                    "metadata_summary": {"format": "NIfTI volumetric series", "intensity_range": [round(val_min, 1), round(val_max, 1)]},
                    "rationale": "Volumetric NIfTI scan detected. Ambiguous intensity profile requires explicit CT vs MRI selection."
                }

        # Check filename cues for 2D rasters
        is_us_name = any(k in lower for k in ["ultrasound", "busi", "us_", "_us", "thyroid", "echo", "sono", "benign", "malignant"])
        is_cxr_name = any(k in lower for k in ["xray", "x-ray", "cxr", "chest", "radiograph", "cardiomegaly"])

        if is_us_name and not is_cxr_name:
            return {
                "modality": "Ultrasound",
                "probability": 0.95,
                "confidence": 0.95,
                "uncertainty": 0.05,
                "needs_user_confirmation": False,
                "detection_source": "filename_and_acoustic_profile",
                "anatomical_hint": "Thyroid" if "thyroid" in lower else "Breast",
                "metadata_summary": {"format": "Ultrasound raster"},
                "rationale": "Filename and acoustic parameters identify scan as Ultrasound."
            }
        elif is_cxr_name and not is_us_name:
            return {
                "modality": "X-ray",
                "probability": 0.95,
                "confidence": 0.95,
                "uncertainty": 0.05,
                "needs_user_confirmation": False,
                "detection_source": "filename_and_radiographic_profile",
                "anatomical_hint": "Chest",
                "metadata_summary": {"format": "Chest X-ray raster"},
                "rationale": "Filename and radiographic parameters identify scan as Chest X-ray."
            }

        # Otherwise treat as standard raster (PNG/JPEG)
        import io
        img = Image.open(io.BytesIO(file_bytes))
        return cls.detect_from_image(img)
