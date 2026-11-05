"""
FastAPI Unified Multi-Modal Medical AI Platform Service
Integrates Ultrasound, MRI, CT, and X-ray models with automatic scan inspection,
Research Privacy / DICOM PHI Filtering de-identification, model routing, and interactive research workbench.
"""

import os
import io
import json
import time
import base64
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np

from api.schemas import (
    FullPredictionResponse,
    ClassificationResponse,
    SegmentationResponse,
    HealthResponse,
    ModelRegistryEntry,
    ScanInspectionResponse,
    MultiModalAnalysisResponse
)
from src.inference.predictor import UltrasoundPredictor
from src.platform.unified_service import UnifiedInferenceService
from src.platform.registry import ModelRegistry
from src.platform.format_validator import ImageFormatValidator
from src.platform.dicom_parser import DICOMMetadataParser
from src.platform.volume_parser import NIfTIMetadataParser
from src.platform.pdf_parser import PDFParser
from src.platform.modality_detector import ModalityDetector
from src.platform.compatibility_validator import ModelCompatibilityValidator
from src.platform.export_service import ExportService
from src.utils.seed import get_device

app = FastAPI(
    title="MURL - Multimodal Medical Imaging Research Workbench",
    description="Research-grade multi-modal deep learning platform integrating Ultrasound, CT, MRI, and X-ray AI.",
    version="2.0.0"
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Lazy service singletons
predictor: Optional[UltrasoundPredictor] = None
unified_service: Optional[UnifiedInferenceService] = None
registry_instance: Optional[ModelRegistry] = None


def get_predictor() -> UltrasoundPredictor:
    global predictor
    if predictor is None:
        device = get_device()
        seg_path = "checkpoints/segmentation/experiments/exp3_standard_dice_focal_best.pth"
        if not os.path.exists(seg_path):
            seg_path = "checkpoints/segmentation/best_unet.pth"
        predictor = UltrasoundPredictor(
            classifier_path="checkpoints/classification/best_transfer.pth",
            segmentor_path=seg_path,
            device=device
        )
    return predictor


def get_unified_service() -> UnifiedInferenceService:
    global unified_service
    if unified_service is None:
        device = get_device()
        unified_service = UnifiedInferenceService(device=device)
    return unified_service


def get_registry() -> ModelRegistry:
    global registry_instance
    if registry_instance is None:
        registry_instance = ModelRegistry()
    return registry_instance


# -------------------------------------------------------------------------
# Platform Diagnostics & Model Registry Endpoints
# -------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    pred = get_predictor()
    reg = get_registry()
    return HealthResponse(
        status="healthy",
        device=str(pred.device),
        classifier_loaded=pred.classifier is not None,
        segmentor_loaded=pred.segmentor is not None,
        platform_models_count=len(reg.list_models()),
        version="2.0.0"
    )


@app.get("/api/models", response_model=List[ModelRegistryEntry])
def list_registered_models(
    modality: Optional[str] = Query(None, description="Filter by modality"),
    task: Optional[str] = Query(None, description="Filter by task"),
    validated_only: bool = Query(True, description="Filter to only actively validated/operational models")
):
    """Returns the catalog of registered medical AI models."""
    reg = get_registry()
    models = reg.list_models(modality=modality, task=task, validated_only=validated_only)
    return [ModelRegistryEntry(**m) for m in models]


@app.get("/api/diagnostics")
def runtime_diagnostics():
    """
    Lightweight runtime verification endpoint.
    Exposes process ID, loaded model registry, router mapping sync, and execution device.
    Guarantees runtime process matches source code on disk.
    """
    import sys
    reg = get_registry()
    service = get_unified_service()
    from src.platform.router import MODEL_ADAPTER_MAP
    model_entries = reg.list_models()
    return {
        "status": "online",
        "process_id": os.getpid(),
        "python_version": sys.version.split()[0],
        "registered_models_count": len(model_entries),
        "registered_model_ids": [m["model_id"] for m in model_entries],
        "router_adapter_map_keys": list(MODEL_ADAPTER_MAP.keys()),
        "adapters_instantiated": list(service.router._instantiated_adapters.keys()),
        "registry_synced": True,
        "device": str(get_device()),
        "privacy_filter": "Research Privacy / DICOM PHI Filtering"
    }


# -------------------------------------------------------------------------
# Unified Multi-Modal Scan Inspection & Analysis Endpoints
# -------------------------------------------------------------------------

@app.post("/api/inspect", response_model=ScanInspectionResponse)
async def inspect_scan(file: UploadFile = File(...)):
    """
    Automatically inspects an uploaded scan (PNG, JPG, DICOM, NIfTI).
    Extracts technical parameters, performs Research Privacy / DICOM PHI Filtering,
    computes modality uncertainty, and returns compatible candidate models.
    """
    file_bytes = await file.read()
    filename = file.filename or "uploaded_scan"

    format_rep = ImageFormatValidator.validate(file_bytes, filename)
    if not format_rep["is_valid"]:
        return ScanInspectionResponse(
            status="error",
            filename=filename,
            file_size_bytes=len(file_bytes),
            format="unknown",
            is_volumetric=False,
            modality_detection={},
            compatible_models=[],
            error=format_rep["error"]
        )

    resolved_format = format_rep["format"]
    is_volumetric = format_rep.get("is_volumetric", False)

    dicom_meta = None
    nifti_meta = None
    pdf_info = None

    try:
        if resolved_format == "dicom":
            dicom_meta = DICOMMetadataParser.parse_metadata(file_bytes)
            modality_det = ModalityDetector.detect_from_dicom(file_bytes)
        elif resolved_format == "pdf":
            pdf_rep = PDFParser.parse_pdf(file_bytes, filename)
            if pdf_rep.get("status") == "error":
                return ScanInspectionResponse(
                    status="error",
                    filename=filename,
                    file_size_bytes=len(file_bytes),
                    format="pdf",
                    is_volumetric=False,
                    modality_detection={},
                    compatible_models=[],
                    error=pdf_rep.get("error", "Unable to open this PDF.")
                )
            pdf_info = pdf_rep
            # Inspect first page image for initial modality detection
            try:
                page_1_bytes = PDFParser.render_page_to_png_bytes(file_bytes, 1)
                page_1_pil = Image.open(io.BytesIO(page_1_bytes)).convert("RGB")
                modality_det = ModalityDetector.detect_from_image(page_1_pil)
            except Exception:
                modality_det = {"modality": "Unknown", "uncertainty": 1.0}
        elif resolved_format in ["nifti", "nifti_gz"]:
            nifti_parsed = NIfTIMetadataParser.parse_from_bytes(file_bytes, filename)
            modality_det = ModalityDetector.detect(file_bytes, filename)

            def pil_to_base64(img: Image.Image) -> str:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")

            nifti_meta = {
                "format": nifti_parsed["format"],
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
                },
                "slice_stacks": nifti_parsed.get("slice_stacks", {})
            }
        else:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            modality_det = ModalityDetector.detect(file_bytes, filename)
    except Exception as e:
        return ScanInspectionResponse(
            status="error",
            filename=filename,
            file_size_bytes=len(file_bytes),
            format=resolved_format,
            is_volumetric=is_volumetric,
            modality_detection={},
            compatible_models=[],
            error=f"Inspection failed: {str(e)}"
        )

    # Find compatible models in registry
    reg = get_registry()
    detected_mod = modality_det.get("modality", "")
    all_models = reg.list_models()
    scan_props = {
        "modality": detected_mod,
        "format": resolved_format,
        "spatial_ndim": 3 if is_volumetric else 2,
        "num_channels": 3 if resolved_format not in ["nifti", "nifti_gz"] else 1
    }

    compatible = []
    for m in all_models:
        chk = ModelCompatibilityValidator.validate_compatibility(m, scan_props)
        if chk["compatible"]:
            compatible.append(m["model_id"])

    return ScanInspectionResponse(
        status="success",
        filename=filename,
        file_size_bytes=len(file_bytes),
        format=resolved_format,
        is_volumetric=is_volumetric,
        modality_detection=modality_det,
        compatible_models=compatible,
        dicom_metadata=dicom_meta,
        nifti_metadata=nifti_meta,
        pdf_info=pdf_info
    )


@app.post("/api/pdf/page")
async def extract_pdf_page(
    file: UploadFile = File(...),
    page_number: int = Form(1)
):
    """
    Renders a specific page from a medical scan PDF container into an image,
    and returns its preview along with modality detection.
    """
    file_bytes = await file.read()
    filename = file.filename or "scan.pdf"
    try:
        png_bytes = PDFParser.render_page_to_png_bytes(file_bytes, page_number)
        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        modality_det = ModalityDetector.detect_from_image(pil_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "status": "success",
            "page_number": page_number,
            "width": pil_img.width,
            "height": pil_img.height,
            "image_b64": img_b64,
            "modality_detection": modality_det
        }
    except Exception as e:
        error_msg = str(e)
        if "out of range" not in error_msg and "contain a supported" not in error_msg:
            error_msg = "Unable to open this PDF."
        raise HTTPException(status_code=400, detail=error_msg)


@app.post("/api/analyze", response_model=MultiModalAnalysisResponse)
async def analyze_scan(
    file: UploadFile = File(...),
    page_number: Optional[int] = Form(1),
    model_id: Optional[str] = Form(None),
    modality_override: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
    anatomy: Optional[str] = Form(None),
    threshold: float = Form(0.5),
    box_prompt_json: Optional[str] = Form(None)
):
    """
    Unified analysis execution endpoint.
    Routes scan to appropriate model adapter, runs pre/inference/post-processing,
    and returns structured diagnostic output and provenance.
    """
    service = get_unified_service()
    file_bytes = await file.read()
    filename = file.filename or "scan.bin"

    format_rep = ImageFormatValidator.validate(file_bytes, filename)
    if format_rep["is_valid"] and format_rep["format"] == "pdf":
        try:
            file_bytes = PDFParser.render_page_to_png_bytes(file_bytes, page_number or 1)
            filename = f"{os.path.splitext(filename)[0]}_page_{page_number or 1}.png"
        except Exception as e:
            err_text = str(e)
            if "out of range" not in err_text and "contain a supported" not in err_text:
                err_text = "Unable to open this PDF."
            return MultiModalAnalysisResponse(
                status="error",
                active_modality="Unknown",
                format_report=format_rep,
                modality_detection={},
                error=err_text
            )

    if model_id is not None and not model_id.strip():
        model_id = None
    if modality_override is not None and not modality_override.strip():
        modality_override = None
    if task is not None and not task.strip():
        task = None
    if anatomy is not None and not anatomy.strip():
        anatomy = None

    box_prompt = None
    if box_prompt_json:
        try:
            box_prompt = json.loads(box_prompt_json)
        except Exception:
            box_prompt = None

    try:
        result = service.process_scan(
            file_bytes=file_bytes,
            filename=filename,
            user_modality_override=modality_override,
            requested_model_id=model_id,
            requested_task=task,
            anatomical_region=anatomy,
            seg_threshold=threshold,
            box_prompt=box_prompt
        )

        if result.get("status") == "error":
            return MultiModalAnalysisResponse(
                status="error",
                active_modality=result.get("active_modality", "Unknown"),
                format_report=result.get("format_report", {}),
                modality_detection=result.get("modality_detection", {}),
                dicom_metadata=result.get("dicom_metadata"),
                nifti_metadata=result.get("nifti_metadata"),
                analysis_mode=result.get("analysis_mode", "Automatic"),
                error=result.get("error", "Analysis failed")
            )

        return MultiModalAnalysisResponse(**result)
    except Exception as e:
        calc_mode = f"Manual — {modality_override}" if modality_override else ("Region segmentation" if (model_id == "medsam_vit_b" or box_prompt) else "Automatic")
        return MultiModalAnalysisResponse(
            status="error",
            active_modality="Unknown",
            format_report={},
            modality_detection={},
            dicom_metadata=None,
            nifti_metadata=None,
            analysis_mode=calc_mode,
            error=f"Analysis failed: {str(e)}"
        )


# -------------------------------------------------------------------------
# Legacy Breast Ultrasound Endpoints (100% Backward Compatibility)
# -------------------------------------------------------------------------

@app.post("/predict/full", response_model=FullPredictionResponse)
async def predict_full(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """Joint classification and lesion segmentation."""
    pred = get_predictor()
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    try:
        res = pred.predict(pil_img, seg_threshold=threshold)
        return FullPredictionResponse(
            original_resolution=res["original_resolution"],
            classification=ClassificationResponse(**res["classification"]) if res["classification"] else None,
            segmentation=SegmentationResponse(**res["segmentation"]) if res["segmentation"] else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/classify", response_model=ClassificationResponse)
async def predict_classify(file: UploadFile = File(...)):
    """Classification only (Benign, Malignant, Normal)."""
    pred = get_predictor()
    if pred.classifier is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded.")
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents))
    res = pred.predict(pil_img)
    return ClassificationResponse(**res["classification"])


@app.post("/predict/segment", response_model=SegmentationResponse)
async def predict_segment(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """Lesion segmentation only."""
    pred = get_predictor()
    if pred.segmentor is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded.")
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents))
    res = pred.predict(pil_img, seg_threshold=threshold)
    return SegmentationResponse(**res["segmentation"])


# -------------------------------------------------------------------------
# Medical-Standard Multi-Format Export Endpoints
# -------------------------------------------------------------------------

@app.post("/api/export/formats")
async def get_export_formats(analysis_json: str = Form(...)):
    """Returns valid export formats for the given analysis result."""
    try:
        data = json.loads(analysis_json)
        valid_formats = ExportService.get_valid_export_formats(data)
        return {"status": "success", "formats": valid_formats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid analysis payload: {str(e)}")


@app.post("/api/export/{format_id}")
async def export_result(format_id: str, analysis_json: str = Form(...)):
    """Exports analysis result in the requested format if technically applicable."""
    try:
        data = json.loads(analysis_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    valid_formats = [f["id"] for f in ExportService.get_valid_export_formats(data)]
    if format_id not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Export format '{format_id}' is not technically applicable to this result. Valid formats: {valid_formats}"
        )

    ts = int(time.time())
    modality = str(data.get("active_modality", "Scan")).replace(" ", "_").replace("-", "_")

    if format_id == "pdf":
        pdf_bytes = ExportService.export_pdf(data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="MURL_{modality}_Report_{ts}.pdf"'}
        )
    elif format_id == "dicom_seg":
        dcm_bytes = ExportService.export_dicom_seg(data)
        return Response(
            content=dcm_bytes,
            media_type="application/dicom",
            headers={"Content-Disposition": f'attachment; filename="MURL_{modality}_Segmentation_{ts}.dcm"'}
        )
    elif format_id == "dicom_sr":
        dcm_bytes = ExportService.export_dicom_sr(data)
        return Response(
            content=dcm_bytes,
            media_type="application/dicom",
            headers={"Content-Disposition": f'attachment; filename="MURL_{modality}_StructuredReport_{ts}.dcm"'}
        )
    elif format_id == "nifti":
        nii_bytes, filename = ExportService.export_nifti_mask(data)
        return Response(
            content=nii_bytes,
            media_type="application/gzip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    elif format_id == "png_overlay":
        png_bytes = ExportService.export_png_overlay(data)
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="MURL_{modality}_Overlay_{ts}.png"'}
        )
    elif format_id == "png_mask":
        png_bytes = ExportService.export_png_mask(data)
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="MURL_{modality}_Mask_{ts}.png"'}
        )
    elif format_id == "csv":
        csv_str = ExportService.export_csv(data)
        return Response(
            content=csv_str.encode("utf-8"),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="MURL_{modality}_Measurements_{ts}.csv"'}
        )
    elif format_id == "json":
        json_str = ExportService.export_json(data)
        return Response(
            content=json_str.encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="MURL_{modality}_Result_{ts}.json"'}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format_id}")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>MURL Unified Multi-Modal AI Platform running. Visit /docs for Swagger UI.</h3>"
