"""
Comprehensive Test Suite for MURL Multi-Modal Medical AI Platform
Tests:
- All 4 Input Formats (PNG, JPG, DICOM, NIfTI 3D/4D)
- DICOM Metadata Extraction and Research Privacy PHI Filtering
- NIfTI 3D & 4D Geometry, Spacing, and Orthogonal Previews
- Modality Uncertainty & Detection
- Model Registry Schema (17 Mandatory Fields)
- Model Compatibility Validator & Router Safety Gates
- Upstream Adapters: MedSAM, TotalSegmentator, nnU-Net, MONAI U-Net, ResNet18
- FastAPI Multi-Modal Endpoints (/api/models, /api/inspect, /api/analyze)
- Preservation of Existing BUSI Endpoints
"""

import io
import os
import json
import pytest
import numpy as np
from PIL import Image
import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
from fastapi.testclient import TestClient

from api.app import app
from src.platform.format_validator import ImageFormatValidator
from src.platform.dicom_parser import DICOMMetadataParser, PHI_TAGS_TO_STRIP
from src.platform.volume_parser import NIfTIMetadataParser
from src.platform.modality_detector import ModalityDetector
from src.platform.registry import ModelRegistry, MANDATORY_REGISTRY_FIELDS
from src.platform.compatibility_validator import ModelCompatibilityValidator
from src.platform.router import ModelRouter
from src.platform.unified_service import UnifiedInferenceService
from src.platform.adapters.medsam_adapter import MedSAMAdapter
from src.platform.adapters.totalsegmentator_adapter import TotalSegmentatorAdapter
from src.platform.adapters.nnunet_adapter import nnUNetAdapter
from src.platform.adapters.chest_xray_adapter import ChestXRayAdapter
from src.platform.adapters.monai_brats_adapter import MONAIBraTSAdapter
from src.platform.adapters.thyroid_ultrasound_adapter import ThyroidUltrasoundAdapter

client = TestClient(app)


def create_synthetic_dicom_bytes(modality="US", patient_name="Secret^Patient", patient_id="PHI-999"):
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = modality
    ds.BodyPartExamined = "BREAST"
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19800101"
    ds.InstitutionName = "Confidential Hospital"
    ds.Rows = 64
    ds.Columns = 64
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = (np.random.rand(64, 64) * 255).astype(np.uint8).tobytes()

    bio = io.BytesIO()
    ds.save_as(bio, enforce_file_format=True)
    return bio.getvalue()


def create_synthetic_nifti_3d_bytes(shape=(32, 32, 16), zooms=(1.5, 1.5, 2.0)):
    arr = (np.random.rand(*shape) * 100).astype(np.float32)
    affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
    img = nib.Nifti1Image(arr, affine)
    bio = io.BytesIO()
    # Save as uncompressed NIfTI
    file_map = img.make_file_map({'image': bio})
    img.to_file_map(file_map)
    return bio.getvalue()


def create_synthetic_nifti_4d_bytes(shape=(16, 16, 8, 4)):
    arr = (np.random.rand(*shape) * 100).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(arr, affine)
    bio = io.BytesIO()
    file_map = img.make_file_map({'image': bio})
    img.to_file_map(file_map)
    return bio.getvalue()


# -------------------------------------------------------------------------
# 1. Format Validation Tests
# -------------------------------------------------------------------------

class TestFormatValidator:
    def test_valid_png_and_magic(self):
        img = Image.new("RGB", (32, 32), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        res = ImageFormatValidator.validate(buf.getvalue(), "scan.png")
        assert res["is_valid"] is True
        assert res["format"] == "png"

    def test_invalid_extension(self):
        res = ImageFormatValidator.validate(b"hello world", "script.exe")
        assert res["is_valid"] is False
        assert "Unsupported file extension" in res["error"]

    def test_corrupted_header_mismatch(self):
        img = Image.new("RGB", (32, 32), color="white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        res = ImageFormatValidator.validate(buf.getvalue(), "scan.png")
        assert res["is_valid"] is False
        assert "header mismatch" in res["error"]

    def test_empty_file_rejected(self):
        res = ImageFormatValidator.validate(b"", "empty.png")
        assert res["is_valid"] is False
        assert "empty" in res["error"]


# -------------------------------------------------------------------------
# 2. DICOM Parsing & Privacy Tests
# -------------------------------------------------------------------------

class TestDICOMParserAndDeIdentification:
    def test_research_privacy_phi_filtering(self):
        dcm_bytes = create_synthetic_dicom_bytes(modality="US", patient_name="John^Doe", patient_id="ID-SECRET")
        metadata = DICOMMetadataParser.parse_metadata(dcm_bytes)
        assert metadata["is_dicom"] is True
        assert metadata["modality"] == "US"
        assert metadata["body_part_examined"] == "BREAST"

        meta_str = str(metadata)
        assert "John^Doe" not in meta_str
        assert "ID-SECRET" not in meta_str
        assert "Confidential Hospital" not in meta_str
        for phi_tag in PHI_TAGS_TO_STRIP:
            assert phi_tag not in metadata

    def test_extract_displayable_image(self):
        dcm_bytes = create_synthetic_dicom_bytes()
        pil_img = DICOMMetadataParser.extract_displayable_image(dcm_bytes)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (64, 64)


# -------------------------------------------------------------------------
# 3. NIfTI 3D & 4D Volumetric Processing Tests
# -------------------------------------------------------------------------

class TestNIfTIParser:
    def test_nifti_3d_inspection(self):
        nii_bytes = create_synthetic_nifti_3d_bytes(shape=(32, 32, 16), zooms=(1.5, 1.5, 2.0))
        meta = NIfTIMetadataParser.parse_from_bytes(nii_bytes, "brain_scan.nii")

        assert meta["format"] == "nifti"
        assert meta["dimensions"] == [32, 32, 16]
        assert meta["spatial_ndim"] == 3
        assert meta["voxel_spacing_mm"] == [1.5, 1.5, 2.0]
        assert "preview_slices" in meta
        assert "axial" in meta["preview_slices"]
        assert "coronal" in meta["preview_slices"]
        assert "sagittal" in meta["preview_slices"]

    def test_nifti_4d_inspection(self):
        nii4d_bytes = create_synthetic_nifti_4d_bytes(shape=(16, 16, 8, 4))
        meta = NIfTIMetadataParser.parse_from_bytes(nii4d_bytes, "functional_mri.nii")

        assert meta["dimensions"] == [16, 16, 8, 4]
        assert meta["spatial_ndim"] == 4
        assert meta["num_channels_or_volumes"] == 4


# -------------------------------------------------------------------------
# 4. Modality Detection & Uncertainty Tests
# -------------------------------------------------------------------------

class TestModalityDetector:
    def test_dicom_modality_detection(self):
        dcm_bytes = create_synthetic_dicom_bytes(modality="US")
        result = ModalityDetector.detect(dcm_bytes, "scan.dcm")
        assert result["modality"] == "Ultrasound"
        assert result["probability"] == 1.0
        assert result["uncertainty"] == 0.0
        assert result["needs_user_confirmation"] is False

    def test_raster_uncertainty_reporting(self):
        img = Image.new("L", (100, 100), color=50)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        res = ModalityDetector.detect(buf.getvalue(), "scan.png")

        assert "probability" in res
        assert "uncertainty" in res
        assert res["uncertainty"] >= 0.0
        assert res["needs_user_confirmation"] is True


# -------------------------------------------------------------------------
# 5. Model Registry & Compatibility Tests
# -------------------------------------------------------------------------

class TestModelRegistry:
    def test_registry_schema_all_17_fields(self):
        registry = ModelRegistry("model_registry/registry.yaml")
        models = registry.list_models()
        assert len(models) >= 5

        for m in models:
            for field in MANDATORY_REGISTRY_FIELDS:
                assert field in m, f"Field '{field}' missing from model '{m.get('model_id')}'"

    def test_only_validated_models_routable_automatically(self):
        registry = ModelRegistry("model_registry/registry.yaml")
        validated = registry.get_validated_models()
        assert len(validated) >= 2
        for vm in validated:
            assert vm["validation_status"] == "validated"

        # Planned models cannot be routed automatically
        assert not registry.is_automatically_routable("monai_lung_nodule_ct_detector")
        assert registry.is_automatically_routable("chexpert_chest_xray_classifier")


class TestModelCompatibilityValidator:
    def test_modality_mismatch_detection(self):
        registry = ModelRegistry("model_registry/registry.yaml")
        meta = registry.get_model("busi_resnet18_classifier")
        bad_scan = {"modality": "MRI", "format": "png", "spatial_ndim": 2, "num_channels": 3}
        res = ModelCompatibilityValidator.validate_compatibility(meta, bad_scan)
        assert res["compatible"] is False
        assert any("Modality mismatch" in s for s in res["issues"])

    def test_dimensionality_mismatch_detection(self):
        registry = ModelRegistry("model_registry/registry.yaml")
        meta = registry.get_model("busi_resnet18_classifier")
        bad_scan = {"modality": "Ultrasound", "format": "png", "spatial_ndim": 3, "num_channels": 3}
        res = ModelCompatibilityValidator.validate_compatibility(meta, bad_scan)
        assert res["compatible"] is False
        assert any("Dimensionality mismatch" in s for s in res["issues"])


# -------------------------------------------------------------------------
# 6. Open-Source Upstream Adapters Tests
# -------------------------------------------------------------------------

class TestUpstreamAdapters:
    def test_medsam_adapter_preprocessing(self):
        config = {
            "model_id": "medsam_vit_b",
            "name": "MedSAM ViT-B",
            "output_task": "interactive_segmentation",
            "weights_path": "checkpoints/medsam/medsam_vit_b.pth"
        }
        adapter = MedSAMAdapter(config)
        dummy_img = Image.new("RGB", (200, 150), color="blue")
        norm, orig_shape = adapter.preprocess(dummy_img)

        assert orig_shape == (200, 150)
        assert norm.shape == (1024, 1024, 3)

        out = adapter.predict(dummy_img, context={"box_prompt": [20, 20, 80, 80]})
        assert out.model_id == "medsam_vit_b"
        assert out.formatted_results["box_prompt_received"] == [20, 20, 80, 80]

    def test_totalsegmentator_adapter_preprocessing(self):
        config = {
            "model_id": "totalsegmentator_ct",
            "name": "TotalSegmentator CT",
            "output_task": "segmentation",
            "weights_path": "checkpoints/ct/totalsegmentator.pth"
        }
        adapter = TotalSegmentatorAdapter(config)
        synthetic_vol = np.zeros((30, 30, 30), dtype=np.float32)
        synthetic_vol[10:20, 10:20, 10:20] = 50.0  # Tissue HU

        out = adapter.predict(synthetic_vol)
        assert out.model_id == "totalsegmentator_ct"
        assert "anatomical_structures_supported" in out.formatted_results

    def test_nnunet_adapter_fingerprinting(self):
        config = {
            "model_id": "nnunet_planner",
            "name": "nnU-Net Planner",
            "output_task": "planning",
            "weights_path": "checkpoints/nnunet/planner.pth"
        }
        adapter = nnUNetAdapter(config)
        arr = np.random.rand(64, 64, 32).astype(np.float32)
        out = adapter.predict(arr)
        assert out.model_id == "nnunet_planner"
        assert "suggested_nnunet_plan" in out.formatted_results["analysis"]

    def test_chest_xray_adapter_inference(self):
        config = {
            "model_id": "chexpert_chest_xray_classifier",
            "name": "TorchXRayVision DenseNet-121",
            "output_task": "classification"
        }
        adapter = ChestXRayAdapter(config)
        assert adapter.is_loaded is True
        cxr_input = "data/samples/nih_chest_xray_00000001.png" if os.path.exists("data/samples/nih_chest_xray_00000001.png") else np.random.randint(40, 220, (224, 224), dtype=np.uint8)
        out = adapter.predict(cxr_input)
        assert out.model_id == "chexpert_chest_xray_classifier"
        assert out.formatted_results["pathologies_evaluated"] == 18
        assert len(out.formatted_results["top_findings"]) == 5
        assert "clinical_disclaimer" in out.formatted_results

    def test_monai_brats_adapter_inference(self):
        config = {
            "model_id": "monai_brats_mri_segmentor",
            "name": "MONAI BraTS MRI Segmentor",
            "output_task": "segmentation"
        }
        adapter = MONAIBraTSAdapter(config)
        assert adapter.is_loaded is True
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        vol[8:24, 8:24, 8:24] = 80.0
        out = adapter.predict(vol)
        assert out.model_id == "monai_brats_mri_segmentor"
        assert "primary_volumes_cm3" in out.formatted_results
        assert out.formatted_results.get("pipeline_status") in ["executed", "checkpoint_required"]
        assert "multiplanar_previews" in out.formatted_results

    def test_thyroid_adapter_tirads_aspect_ratio(self):
        config = {
            "model_id": "tn3k_thyroid_segmentor",
            "name": "Thyroid Acoustic U-Net",
            "output_task": "segmentation"
        }
        adapter = ThyroidUltrasoundAdapter(config)
        assert adapter.is_loaded is True
        img = np.zeros((200, 200), dtype=np.uint8)
        img[40:160, 80:120] = 200
        out = adapter.predict(img)
        assert out.model_id == "tn3k_thyroid_segmentor"
        assert "tirads_features" in out.formatted_results
        assert "mask_overlay_b64" in out.formatted_results

    def test_anatomical_isolation_breast_vs_thyroid(self):
        registry = ModelRegistry("model_registry/registry.yaml")
        breast_model = registry.get_model("busi_resnet18_classifier")
        thyroid_scan = {"modality": "Ultrasound", "anatomical_region": "Thyroid", "format": "png", "spatial_ndim": 2, "num_channels": 3}
        res_breast = ModelCompatibilityValidator.validate_compatibility(breast_model, thyroid_scan)
        assert res_breast["compatible"] is False
        assert any("Anatomical region mismatch" in s for s in res_breast["issues"])

        thyroid_model = registry.get_model("tn3k_thyroid_segmentor")
        breast_scan = {"modality": "Ultrasound", "anatomical_region": "Breast", "format": "png", "spatial_ndim": 2, "num_channels": 1}
        res_thyroid = ModelCompatibilityValidator.validate_compatibility(thyroid_model, breast_scan)
        assert res_thyroid["compatible"] is False
        assert any("Anatomical region mismatch" in s for s in res_thyroid["issues"])


# -------------------------------------------------------------------------
# 7. FastAPI Multi-Modal Endpoints Tests
# -------------------------------------------------------------------------

class TestFastAPIEndpoints:
    def test_health_v2(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "2.0.0"
        assert data["platform_models_count"] >= 5

    def test_api_models_endpoint(self):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 5
        model_ids = [m["model_id"] for m in data]
        assert "busi_resnet18_classifier" in model_ids
        assert "medsam_vit_b" in model_ids
        assert "totalsegmentator_ct" in model_ids

    def test_api_inspect_png(self):
        img = Image.new("RGB", (64, 64), color="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post("/api/inspect", files={"file": ("scan.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["format"] == "png"
        assert "modality_detection" in data

    def test_api_inspect_nifti(self):
        nii_bytes = create_synthetic_nifti_3d_bytes()
        resp = client.post("/api/inspect", files={"file": ("vol.nii", nii_bytes, "application/octet-stream")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["format"] == "nifti"
        assert data["is_volumetric"] is True
        assert data["nifti_metadata"] is not None

    def test_api_analyze_validated_model(self):
        img = Image.new("RGB", (64, 64), color="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/api/analyze",
            files={"file": ("us.png", buf.getvalue(), "image/png")},
            data={"model_id": "busi_resnet18_classifier"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["prediction"] is not None
        assert "formatted_results" in data["prediction"]

    def test_api_analyze_chest_xray(self):
        if os.path.exists("data/samples/nih_chest_xray_00000001.png"):
            with open("data/samples/nih_chest_xray_00000001.png", "rb") as f:
                cxr_bytes = f.read()
        else:
            cxr = np.random.randint(40, 220, (224, 224), dtype=np.uint8)
            img = Image.fromarray(cxr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            cxr_bytes = buf.getvalue()

        resp = client.post(
            "/api/analyze",
            files={"file": ("cxr.png", cxr_bytes, "image/png")},
            data={"model_id": "chexpert_chest_xray_classifier"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["prediction"]["model_id"] == "chexpert_chest_xray_classifier"
        assert "top_findings" in data["prediction"]["formatted_results"]
        assert len(data["prediction"]["formatted_results"]["top_findings"]) == 5

    def test_api_analyze_blocks_unvalidated_model(self):
        img = Image.new("RGB", (64, 64), color="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/api/analyze",
            files={"file": ("us.png", buf.getvalue(), "image/png")},
            data={"model_id": "monai_lung_nodule_ct_detector"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "planned" in data["error"]

    def test_legacy_predict_full_preserved(self):
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post("/predict/full", files={"file": ("test.png", buf.getvalue(), "image/png")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "classification" in data
        assert "segmentation" in data
