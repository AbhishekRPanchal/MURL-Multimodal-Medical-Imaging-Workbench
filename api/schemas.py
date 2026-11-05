"""
Pydantic Schemas for Multi-Modal Medical AI REST API
Preserves existing ultrasound schemas while introducing multi-modal platform models.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


# -------------------------------------------------------------------------
# Legacy Ultrasound Schemas (100% Backward-Compatible)
# -------------------------------------------------------------------------

class ClassificationResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted diagnosis: benign, malignant, or normal")
    confidence: float = Field(..., description="Model softmax confidence score [0.0 - 1.0]")
    probabilities: Dict[str, float] = Field(..., description="Multi-class posterior probabilities")


class BoundingBox(BaseModel):
    box_xyxy: List[int] = Field(..., description="Bounding box in [x1, y1, x2, y2] format")
    box_xywh: List[int] = Field(..., description="Bounding box in [x, y, w, h] format")
    area_pixels: int = Field(..., description="Total pixel area of the lesion component")


class SegmentationResponse(BaseModel):
    lesion_detected: bool = Field(..., description="True if positive lesion pixels detected")
    lesion_area_percentage: float = Field(..., description="Percentage of ultrasound scan covered by lesion")
    lesion_pixel_count: int = Field(..., description="Count of positive lesion mask pixels")
    derived_bounding_boxes: List[BoundingBox] = Field(..., description="Derived spatial ROIs computed from masks")
    mask_shape: List[int] = Field(..., description="Dimensions [H, W] of the mask")


class FullPredictionResponse(BaseModel):
    status: str = "success"
    original_resolution: List[int] = Field(..., description="Original image dimensions [Width, Height]")
    classification: Optional[ClassificationResponse] = None
    segmentation: Optional[SegmentationResponse] = None
    disclaimer: str = (
        "RESEARCH DEMONSTRATION ONLY. This system is developed for academic evaluation "
        "and is NOT approved for clinical diagnosis or medical decision-making."
    )


class HealthResponse(BaseModel):
    status: str
    device: str
    classifier_loaded: bool
    segmentor_loaded: bool
    platform_models_count: int = 0
    version: str = "2.0.0"


# -------------------------------------------------------------------------
# Multi-Modal Platform Schemas
# -------------------------------------------------------------------------

class ModelRegistryEntry(BaseModel):
    model_id: str
    name: str
    modality: str
    anatomical_region: str
    supported_image_formats: List[str]
    required_input_dimensions: List[int]
    num_input_channels: int
    required_mri_sequences: List[str]
    output_task: str
    dataset: str
    weights_path: str
    source_url: str
    license: str
    training_provenance: str
    evaluation_metrics: Dict[str, Any]
    validation_status: str
    known_limitations: str


class ScanInspectionResponse(BaseModel):
    status: str
    filename: str
    file_size_bytes: int
    format: str
    is_volumetric: bool
    modality_detection: Dict[str, Any]
    compatible_models: List[str]
    dicom_metadata: Optional[Dict[str, Any]] = None
    nifti_metadata: Optional[Dict[str, Any]] = None
    pdf_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MultiModalAnalysisResponse(BaseModel):
    status: str
    active_modality: str
    format_report: Dict[str, Any]
    modality_detection: Dict[str, Any]
    compatibility_report: Optional[Dict[str, Any]] = None
    prediction: Optional[Dict[str, Any]] = None
    dicom_metadata: Optional[Dict[str, Any]] = None
    nifti_metadata: Optional[Dict[str, Any]] = None
    analysis_mode: Optional[str] = "Automatic"
    regulatory_notice: str = (
        "This platform is intended for research and educational purposes. "
        "Outputs are experimental and are not a medical diagnosis or a substitute for qualified clinical interpretation."
    )
    error: Optional[str] = None
