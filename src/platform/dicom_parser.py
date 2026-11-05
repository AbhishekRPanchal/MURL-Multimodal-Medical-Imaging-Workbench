"""
DICOM Metadata Parser with Research Privacy / DICOM PHI Filtering
Extracts technical imaging tags, handles CT windowing and multi-frame datasets,
and enforces patient privacy standards.
"""

import io
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.dataset import FileDataset
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


# Protected Health Information (PHI) tags that are redacted for research privacy filtering
PHI_TAGS_TO_STRIP = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "AccessionNumber",
    "StudyID",
    "OtherPatientIDs",
    "EthnicGroup",
    "MedicalRecordLocator"
]


class DICOMMetadataParser:
    """
    Parses DICOM byte streams or files, extracting technical image parameters
    while guaranteeing de-identification of sensitive clinical patient identifiers.
    """

    @staticmethod
    def parse_metadata(file_source: Any) -> Dict[str, Any]:
        """
        Parses DICOM tags and returns sanitized clinical/technical metadata.
        """
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is not installed. Install pydicom to parse DICOM files.")

        if isinstance(file_source, bytes):
            dataset: FileDataset = pydicom.dcmread(io.BytesIO(file_source), stop_before_pixels=True, force=True)
        elif isinstance(file_source, io.BytesIO):
            file_source.seek(0)
            dataset = pydicom.dcmread(file_source, stop_before_pixels=True, force=True)
        elif isinstance(file_source, str):
            dataset = pydicom.dcmread(file_source, stop_before_pixels=True, force=True)
        else:
            raise TypeError("Unsupported file source for DICOM reading.")

        def safe_get(tag_name: str, default: Any = None) -> Any:
            try:
                val = getattr(dataset, tag_name, default)
                if val is None or val == "":
                    return default
                return str(val).strip()
            except Exception:
                return default

        rows = getattr(dataset, "Rows", None)
        cols = getattr(dataset, "Columns", None)
        num_frames = getattr(dataset, "NumberOfFrames", 1)
        try:
            num_frames = int(num_frames)
        except Exception:
            num_frames = 1

        pixel_spacing = getattr(dataset, "PixelSpacing", None)
        if pixel_spacing:
            try:
                pixel_spacing = [float(x) for x in pixel_spacing]
            except Exception:
                pixel_spacing = str(pixel_spacing)

        slice_thickness = getattr(dataset, "SliceThickness", None)
        if slice_thickness:
            try:
                slice_thickness = float(slice_thickness)
            except Exception:
                slice_thickness = str(slice_thickness)

        window_center = getattr(dataset, "WindowCenter", None)
        window_width = getattr(dataset, "WindowWidth", None)
        if window_center is not None and hasattr(window_center, "__iter__") and not isinstance(window_center, str):
            window_center = float(window_center[0])
        elif window_center is not None:
            try:
                window_center = float(window_center)
            except Exception:
                window_center = None

        if window_width is not None and hasattr(window_width, "__iter__") and not isinstance(window_width, str):
            window_width = float(window_width[0])
        elif window_width is not None:
            try:
                window_width = float(window_width)
            except Exception:
                window_width = None

        sanitized_metadata = {
            "is_dicom": True,
            "modality": safe_get("Modality", "UNKNOWN").upper(),
            "body_part_examined": safe_get("BodyPartExamined", "UNSPECIFIED").upper(),
            "study_description": safe_get("StudyDescription", "DE-IDENTIFIED STUDY"),
            "series_description": safe_get("SeriesDescription", "DE-IDENTIFIED SERIES"),
            "image_type": list(getattr(dataset, "ImageType", [])) if hasattr(dataset, "ImageType") else [],
            "dimensions": {
                "rows": int(rows) if rows is not None else None,
                "columns": int(cols) if cols is not None else None,
                "num_frames": num_frames
            },
            "pixel_spacing_mm": pixel_spacing,
            "slice_thickness_mm": slice_thickness,
            "windowing": {
                "center": window_center,
                "width": window_width
            },
            "sop_class_uid": safe_get("SOPClassUID"),
            "sop_instance_uid": safe_get("SOPInstanceUID"),
            "study_instance_uid": safe_get("StudyInstanceUID"),
            "series_instance_uid": safe_get("SeriesInstanceUID"),
            "frame_of_reference_uid": safe_get("FrameOfReferenceUID"),
            "photometric_interpretation": safe_get("PhotometricInterpretation", "MONOCHROME2"),
            "manufacturer": safe_get("Manufacturer", "ANONYMIZED"),
            "manufacturer_model": safe_get("ManufacturerModelName", "ANONYMIZED"),
            "de_identification_status": "Research privacy filtering applied",
            "phi_stripped": PHI_TAGS_TO_STRIP
        }

        return sanitized_metadata

    @classmethod
    def extract_pixel_array(cls, file_source: Any) -> np.ndarray:
        """
        Extracts raw and rescaled pixel array from DICOM file.
        """
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is not installed.")

        if isinstance(file_source, bytes):
            dataset = pydicom.dcmread(io.BytesIO(file_source), force=True)
        elif isinstance(file_source, io.BytesIO):
            file_source.seek(0)
            dataset = pydicom.dcmread(file_source, force=True)
        elif isinstance(file_source, str):
            dataset = pydicom.dcmread(file_source, force=True)
        else:
            raise TypeError("Unsupported file source.")

        pixel_array = dataset.pixel_array

        slope = getattr(dataset, "RescaleSlope", 1.0)
        intercept = getattr(dataset, "RescaleIntercept", 0.0)
        if slope != 1.0 or intercept != 0.0:
            pixel_array = pixel_array.astype(np.float32) * float(slope) + float(intercept)

        return pixel_array

    @classmethod
    def extract_displayable_image(cls, file_source: Any, frame_idx: int = 0) -> Image.Image:
        """
        Extracts displayable 8-bit RGB PIL Image from DICOM with appropriate windowing.
        """
        pixels = cls.extract_pixel_array(file_source)

        # Handle multi-frame (3D) DICOM volumes
        if pixels.ndim == 3 and pixels.shape[0] > 1 and pixels.shape[-1] not in (3, 4):
            # Shape is (Frames, Rows, Cols)
            frame_idx = min(max(0, frame_idx), pixels.shape[0] - 1)
            slice_2d = pixels[frame_idx]
        elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            # Already RGB/RGBA
            slice_2d = pixels
        else:
            slice_2d = pixels

        if slice_2d.ndim == 3 and slice_2d.shape[-1] in (3, 4):
            return Image.fromarray(slice_2d.astype(np.uint8)).convert("RGB")

        # Grayscale windowing
        p_min, p_max = np.percentile(slice_2d, 1), np.percentile(slice_2d, 99)
        if p_max - p_min > 0:
            norm = np.clip((slice_2d - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            norm = np.zeros_like(slice_2d, dtype=np.uint8)

        return Image.fromarray(norm).convert("RGB")
