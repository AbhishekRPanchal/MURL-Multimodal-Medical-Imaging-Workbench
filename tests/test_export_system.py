"""
Comprehensive Test Suite for MURL Medical-Standard Result Export System
Validates:
1. Context-Aware Format Filtering (only technically valid export options exposed)
2. DICOM SEG (Segmentation Storage SOP Class 1.2.840.10008.5.1.4.1.1.66.4, Modality SEG, SegmentSequence)
3. DICOM SR (Structured Report SOP Class 1.2.840.10008.5.1.4.1.1.88.33, Modality SR, ConceptName, UCUM units)
4. NIfTI Mask (.nii.gz: geometry, affine, dimensions, voxel spacing)
5. PDF Report (ReportLab generation, pypdf validation, disclaimer check, clean operator presentation)
6. PNG Overlay & PNG Mask (2D image integrity and alpha blending)
7. CSV Tabular Metrics (accurate columns and values for CT/MRI/Ultrasound)
8. API Endpoints (/api/export/formats, /api/export/{format_id}) and Safeguard Gates
"""

import io
import json
import csv
import pytest
import numpy as np
from PIL import Image
import nibabel as nib
import pydicom
from pypdf import PdfReader
from fastapi.testclient import TestClient

from api.app import app
from src.platform.export_service import ExportService

client = TestClient(app)


# -------------------------------------------------------------------------
# Fixtures for Different Modalities and Task Outputs
# -------------------------------------------------------------------------

@pytest.fixture
def cxr_classification_result():
    return {
        "status": "success",
        "active_modality": "X-ray",
        "format_report": {
            "format": "png",
            "filename": "nih_cxr_cardiomegaly_01.png",
            "dimensions": [224, 224]
        },
        "prediction": {
            "model_id": "chexpert_chest_xray_classifier",
            "output_task": "classification",
            "status": "success",
            "inference_time_ms": 2500,
            "execution_device": "CPU",
            "formatted_results": {
                "pathologies_evaluated": 18,
                "top_findings": [
                    {"pathology": "Cardiomegaly", "probability_percent": 64.78},
                    {"pathology": "Infiltration", "probability_percent": 24.12},
                    {"pathology": "Effusion", "probability_percent": 18.35}
                ]
            }
        }
    }


@pytest.fixture
def busi_classification_result():
    return {
        "status": "success",
        "active_modality": "Ultrasound",
        "format_report": {
            "format": "png",
            "filename": "busi_benign_01.png",
            "dimensions": [256, 256]
        },
        "prediction": {
            "model_id": "busi_resnet18_classifier",
            "output_task": "classification",
            "status": "success",
            "inference_time_ms": 500,
            "execution_device": "CPU",
            "formatted_results": {
                "predicted_class": "benign",
                "confidence": 98.45
            }
        }
    }


@pytest.fixture
def busi_segmentation_result():
    # Synthetic mask base64
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    m_pil = Image.fromarray(mask)
    buf_m = io.BytesIO()
    m_pil.save(buf_m, format="PNG")
    m_b64 = buf_m.getvalue()
    import base64
    mask_b64 = base64.b64encode(m_b64).decode("utf-8")

    return {
        "status": "success",
        "active_modality": "Ultrasound",
        "format_report": {
            "format": "png",
            "filename": "busi_malignant_01.png",
            "dimensions": [100, 100]
        },
        "prediction": {
            "model_id": "busi_monai_unet_segmentor",
            "output_task": "segmentation",
            "status": "success",
            "inference_time_ms": 120,
            "execution_device": "CPU",
            "formatted_results": {
                "lesion_detected": True,
                "lesion_area_percentage": 25.0,
                "lesion_pixel_count": 2500,
                "mask_shape": [100, 100],
                "raw_mask_b64": mask_b64,
                "mask_overlay_b64": mask_b64,
                "derived_bounding_boxes": [
                    {"box_xywh": [25, 25, 50, 50], "area_pixels": 2500}
                ]
            }
        }
    }


@pytest.fixture
def ct_totalseg_result(tmp_path):
    # Create synthetic NIfTI mask file
    mask_data = np.zeros((30, 30, 20), dtype=np.uint8)
    mask_data[5:15, 5:15, 5:15] = 5  # Liver
    mask_data[18:25, 18:25, 5:15] = 1  # Spleen
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    nii = nib.Nifti1Image(mask_data, affine)
    nii_path = str(tmp_path / "totalseg_test.nii.gz")
    nib.save(nii, nii_path)

    # 2D preview slice base64
    preview_img = Image.fromarray((mask_data[:, :, 10] * 40).astype(np.uint8))
    buf = io.BytesIO()
    preview_img.save(buf, format="PNG")
    import base64
    prev_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "active_modality": "CT",
        "format_report": {
            "format": "nifti_gz",
            "filename": "example_ct_sm.nii.gz",
            "dimensions": [30, 30, 20]
        },
        "prediction": {
            "model_id": "totalsegmentator_ct",
            "output_task": "segmentation",
            "status": "success",
            "inference_time_ms": 28000,
            "execution_device": "CPU",
            "formatted_results": {
                "pipeline_status": "executed",
                "output_nifti_path": nii_path,
                "structures_detected_count": 2,
                "top_detected_structures": [
                    {"structure_name": "liver", "label_id": 5, "voxel_count": 1000, "volume_cm3": 27.0},
                    {"structure_name": "spleen", "label_id": 1, "voxel_count": 490, "volume_cm3": 13.23}
                ],
                "all_detected_structures": [
                    {"structure_name": "liver", "label_id": 5, "voxel_count": 1000, "volume_cm3": 27.0},
                    {"structure_name": "spleen", "label_id": 1, "voxel_count": 490, "volume_cm3": 13.23}
                ],
                "multilabel_previews": {
                    "axial_seg_b64": prev_b64
                }
            }
        }
    }


@pytest.fixture
def mri_brats_result(tmp_path):
    mask_data = np.zeros((32, 32, 24), dtype=np.uint8)
    mask_data[8:24, 8:24, 6:18] = 2  # Edema
    mask_data[12:20, 12:20, 8:16] = 1  # Core
    mask_data[14:18, 14:18, 10:14] = 3  # Enhancing
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nii = nib.Nifti1Image(mask_data, affine)
    nii_path = str(tmp_path / "brats_test.nii.gz")
    nib.save(nii, nii_path)

    preview_img = Image.fromarray((mask_data[:, :, 12] * 60).astype(np.uint8))
    buf = io.BytesIO()
    preview_img.save(buf, format="PNG")
    import base64
    prev_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "active_modality": "MRI",
        "format_report": {
            "format": "nifti_gz",
            "filename": "brats_brain_mri_demo.nii.gz",
            "dimensions": [32, 32, 24]
        },
        "prediction": {
            "model_id": "monai_brats_mri_segmentor",
            "output_task": "segmentation",
            "status": "success",
            "inference_time_ms": 450,
            "execution_device": "CPU",
            "formatted_results": {
                "pipeline_status": "executed",
                "is_trained_ai_segmentation": True,
                "output_nifti_path": nii_path,
                "tumor_detected": True,
                "compartments": [
                    {"compartment": "Whole Tumor (WT)", "description": "Abnormal region", "voxel_count": 3072, "volume_cm3": 3.072},
                    {"compartment": "Tumor Core (TC)", "description": "Gross tumor volume", "voxel_count": 512, "volume_cm3": 0.512},
                    {"compartment": "Enhancing Tumor (ET)", "description": "Active hypervascularized rim", "voxel_count": 64, "volume_cm3": 0.064}
                ],
                "mask_overlay_b64": prev_b64
            }
        }
    }


# -------------------------------------------------------------------------
# 1. Test Context-Aware Format Filtering (Section 1 & 9)
# -------------------------------------------------------------------------

class TestContextAwareFormatFiltering:
    def test_cxr_classification_formats(self, cxr_classification_result):
        formats = ExportService.get_valid_export_formats(cxr_classification_result)
        format_ids = [f["id"] for f in formats]

        assert "pdf" in format_ids
        assert "json" in format_ids
        assert "dicom_sr" in format_ids
        # Strictly disallowed for classification-only
        assert "csv" not in format_ids
        assert "dicom_seg" not in format_ids
        assert "nifti" not in format_ids

    def test_busi_classification_formats(self, busi_classification_result):
        formats = ExportService.get_valid_export_formats(busi_classification_result)
        format_ids = [f["id"] for f in formats]

        assert "pdf" in format_ids
        assert "json" in format_ids
        # Strictly disallowed
        assert "csv" not in format_ids
        assert "dicom_seg" not in format_ids
        assert "nifti" not in format_ids

    def test_busi_segmentation_formats(self, busi_segmentation_result):
        formats = ExportService.get_valid_export_formats(busi_segmentation_result)
        format_ids = [f["id"] for f in formats]

        assert "pdf" in format_ids
        assert "json" in format_ids
        assert "dicom_sr" in format_ids
        assert "dicom_seg" in format_ids
        assert "nifti" in format_ids
        assert "png_overlay" in format_ids
        assert "png_mask" in format_ids
        assert "csv" in format_ids

    def test_ct_segmentation_formats(self, ct_totalseg_result):
        formats = ExportService.get_valid_export_formats(ct_totalseg_result)
        format_ids = [f["id"] for f in formats]

        assert "pdf" in format_ids
        assert "json" in format_ids
        assert "dicom_sr" in format_ids
        assert "dicom_seg" in format_ids
        assert "nifti" in format_ids
        assert "png_overlay" in format_ids
        assert "csv" in format_ids

    def test_mri_segmentation_formats(self, mri_brats_result):
        formats = ExportService.get_valid_export_formats(mri_brats_result)
        format_ids = [f["id"] for f in formats]

        assert "pdf" in format_ids
        assert "json" in format_ids
        assert "dicom_sr" in format_ids
        assert "dicom_seg" in format_ids
        assert "nifti" in format_ids
        assert "png_overlay" in format_ids
        assert "csv" in format_ids


# -------------------------------------------------------------------------
# 2. Test DICOM SEG Genuine Standard Object (Section 2)
# -------------------------------------------------------------------------

class TestDICOMSEGExport:
    def test_ct_dicom_seg(self, ct_totalseg_result):
        seg_bytes = ExportService.export_dicom_seg(ct_totalseg_result)
        assert len(seg_bytes) > 0

        # Parse with genuine pydicom reader
        ds = pydicom.dcmread(io.BytesIO(seg_bytes))
        assert ds.Modality == "SEG"
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
        assert ds.SegmentationType == "BINARY"
        assert hasattr(ds, "SegmentSequence")
        assert len(ds.SegmentSequence) == 2
        assert ds.SegmentSequence[0].SegmentLabel == "liver"
        assert ds.SegmentSequence[1].SegmentLabel == "spleen"
        assert ds.Rows == 30
        assert ds.Columns == 30
        assert ds.NumberOfFrames == 20

    def test_mri_dicom_seg(self, mri_brats_result):
        seg_bytes = ExportService.export_dicom_seg(mri_brats_result)
        ds = pydicom.dcmread(io.BytesIO(seg_bytes))
        assert ds.Modality == "SEG"
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
        assert len(ds.SegmentSequence) == 3
        labels = [s.SegmentLabel for s in ds.SegmentSequence]
        assert "Whole Tumor (WT)" in labels
        assert "Tumor Core (TC)" in labels
        assert "Enhancing Tumor (ET)" in labels

    def test_ultrasound_dicom_seg_2d(self, busi_segmentation_result):
        seg_bytes = ExportService.export_dicom_seg(busi_segmentation_result)
        ds = pydicom.dcmread(io.BytesIO(seg_bytes))
        assert ds.Modality == "SEG"
        assert ds.Rows == 100
        assert ds.Columns == 100
        assert ds.NumberOfFrames == 1
        assert len(ds.SegmentSequence) >= 1

    def test_preserves_source_dicom_references_when_present(self, busi_segmentation_result):
        # Inject source DICOM context
        busi_segmentation_result["dicom_metadata"] = {
            "study_instance_uid": "1.2.3.4.5.6.7",
            "series_instance_uid": "1.2.3.4.5.6.8",
            "sop_instance_uid": "1.2.3.4.5.6.9",
            "patient_id": "ANON-PT-042",
            "patient_name": "ANON^PATIENT"
        }
        seg_bytes = ExportService.export_dicom_seg(busi_segmentation_result)
        ds = pydicom.dcmread(io.BytesIO(seg_bytes))
        assert ds.StudyInstanceUID == "1.2.3.4.5.6.7"
        assert ds.PatientID == "ANON-PT-042"
        assert hasattr(ds, "ReferencedSeriesSequence")
        assert ds.ReferencedSeriesSequence[0].SeriesInstanceUID == "1.2.3.4.5.6.8"


# -------------------------------------------------------------------------
# 3. Test DICOM SR Genuine Structured Report (Section 4)
# -------------------------------------------------------------------------

class TestDICOMStructuredReport:
    def test_cxr_dicom_sr(self, cxr_classification_result):
        sr_bytes = ExportService.export_dicom_sr(cxr_classification_result)
        assert len(sr_bytes) > 0

        ds = pydicom.dcmread(io.BytesIO(sr_bytes))
        assert ds.Modality == "SR"
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.33"
        assert ds.ValueType == "CONTAINER"
        assert ds.ConceptNameCodeSequence[0].CodeMeaning == "Imaging Measurement Report"

        # Content sequence contains summary and quantitative findings
        items = ds.ContentSequence
        assert len(items) >= 4  # summary, procedure, + 3 findings
        pathology_meanings = [item.ConceptNameCodeSequence[0].CodeMeaning for item in items[2:]]
        assert "Cardiomegaly" in pathology_meanings
        assert items[2].MeasuredValueSequence[0].NumericValue == "64.78"
        assert items[2].MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == "%"

    def test_ct_dicom_sr_volumetry(self, ct_totalseg_result):
        sr_bytes = ExportService.export_dicom_sr(ct_totalseg_result)
        ds = pydicom.dcmread(io.BytesIO(sr_bytes))
        assert ds.Modality == "SR"
        items = ds.ContentSequence
        struct_names = [item.ConceptNameCodeSequence[0].CodeMeaning for item in items if item.ValueType == "NUM"]
        assert "liver" in struct_names
        assert "spleen" in struct_names


# -------------------------------------------------------------------------
# 4. Test NIfTI Mask Export (Section 3)
# -------------------------------------------------------------------------

class TestNIfTIMaskExport:
    def test_ct_nifti_mask_geometry(self, ct_totalseg_result):
        nii_bytes, filename = ExportService.export_nifti_mask(ct_totalseg_result)
        assert filename.endswith(".nii.gz")
        assert len(nii_bytes) > 0

        # Load back with nibabel from gzipped bytes
        import gzip
        decompressed = gzip.decompress(nii_bytes)
        loaded_nii = nib.Nifti1Image.from_bytes(decompressed)
        assert loaded_nii.shape == (30, 30, 20)
        assert np.allclose(loaded_nii.header.get_zooms()[:3], (3.0, 3.0, 3.0))

    def test_mri_nifti_mask_geometry(self, mri_brats_result):
        nii_bytes, filename = ExportService.export_nifti_mask(mri_brats_result)
        import gzip
        decompressed = gzip.decompress(nii_bytes)
        loaded_nii = nib.Nifti1Image.from_bytes(decompressed)
        assert loaded_nii.shape == (32, 32, 24)


# -------------------------------------------------------------------------
# 5. Test PDF Report Generation & Disclaimer (Section 5)
# -------------------------------------------------------------------------

class TestPDFReportExport:
    def test_pdf_structure_and_disclaimer(self, cxr_classification_result):
        pdf_bytes = ExportService.export_pdf(cxr_classification_result)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF-1.")

        # Read back with pypdf
        reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) >= 1
        page_text = reader.pages[0].extract_text()

        # Required neutral project branding and document identity
        assert "MURL" in page_text
        assert "Multimodal Medical Imaging Research Workbench" in page_text
        assert "Medical Imaging Analysis Report" in page_text
        assert "Research Privacy Filter Applied" in page_text
        assert "Cardiomegaly" in page_text

        # Verify strict document metadata
        meta = reader.metadata
        if meta:
            assert str(meta.get("/Author", "")) == "MURL Research Workbench"
            assert str(meta.get("/Creator", "")) == "MURL"

        # Required research disclaimer
        assert "RESEARCH DEMONSTRATION ONLY" in page_text
        assert "NOT FOR CLINICAL DIAGNOSIS" in page_text

        # Operator cleanliness: technical internal implementation strings hidden
        assert "chexpert_chest_xray_classifier" not in page_text
        assert "checkpoint_final.pth" not in page_text

    def test_ct_pdf_contains_volumetry_table(self, ct_totalseg_result):
        pdf_bytes = ExportService.export_pdf(ct_totalseg_result)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = reader.pages[0].extract_text()
        assert "Liver" in text
        assert "Spleen" in text
        assert "27.00" in text  # Volume cm3


# -------------------------------------------------------------------------
# 6. Test CSV Metrics Export (Section 8)
# -------------------------------------------------------------------------

class TestCSVExport:
    def test_ct_csv_structure(self, ct_totalseg_result):
        csv_str = ExportService.export_csv(ct_totalseg_result)
        reader = list(csv.reader(io.StringIO(csv_str)))
        assert reader[0] == ["structure_name", "label_id", "voxel_count", "volume_cm3"]
        assert reader[1] == ["liver", "5", "1000", "27.0"]
        assert reader[2] == ["spleen", "1", "490", "13.23"]

    def test_mri_csv_compartments(self, mri_brats_result):
        csv_str = ExportService.export_csv(mri_brats_result)
        reader = list(csv.reader(io.StringIO(csv_str)))
        assert reader[0] == ["compartment", "description", "voxel_count", "volume_cm3"]
        assert "Whole Tumor (WT)" in reader[1][0]

    def test_ultrasound_csv_measurements(self, busi_segmentation_result):
        csv_str = ExportService.export_csv(busi_segmentation_result)
        reader = list(csv.reader(io.StringIO(csv_str)))
        assert reader[0] == ["measurement", "value", "unit"]
        assert reader[1] == ["lesion_detected", "True", "boolean"]
        assert reader[2] == ["lesion_area_percentage", "25.0", "%"]

    def test_classification_rejects_csv(self, cxr_classification_result):
        with pytest.raises(ValueError, match="No meaningful quantitative tabular measurements"):
            ExportService.export_csv(cxr_classification_result)


# -------------------------------------------------------------------------
# 7. Test PNG Overlay and Mask (Section 7)
# -------------------------------------------------------------------------

class TestPNGOverlayAndMask:
    def test_png_overlay(self, busi_segmentation_result):
        overlay_bytes = ExportService.export_png_overlay(busi_segmentation_result)
        img = Image.open(io.BytesIO(overlay_bytes))
        assert img.format == "PNG"
        assert img.size == (100, 100)

    def test_png_mask(self, busi_segmentation_result):
        mask_bytes = ExportService.export_png_mask(busi_segmentation_result)
        img = Image.open(io.BytesIO(mask_bytes))
        assert img.format == "PNG"
        assert img.size == (100, 100)


# -------------------------------------------------------------------------
# 8. Test REST API Endpoints and Safeguards (Section 12, 13)
# -------------------------------------------------------------------------

class TestExportAPIEndpoints:
    def test_api_export_formats_endpoint(self, ct_totalseg_result):
        resp = client.post("/api/export/formats", data={"analysis_json": json.dumps(ct_totalseg_result)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        format_ids = [f["id"] for f in data["formats"]]
        assert "pdf" in format_ids
        assert "dicom_seg" in format_ids
        assert "dicom_sr" in format_ids
        assert "nifti" in format_ids
        assert "csv" in format_ids
        assert "json" in format_ids

    def test_api_export_pdf_download(self, cxr_classification_result):
        resp = client.post("/api/export/pdf", data={"analysis_json": json.dumps(cxr_classification_result)})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert "attachment; filename=" in resp.headers["content-disposition"]
        assert resp.content.startswith(b"%PDF-1.")

    def test_api_export_dicom_seg_download(self, ct_totalseg_result):
        resp = client.post("/api/export/dicom_seg", data={"analysis_json": json.dumps(ct_totalseg_result)})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/dicom"
        ds = pydicom.dcmread(io.BytesIO(resp.content))
        assert ds.Modality == "SEG"

    def test_api_export_dicom_sr_download(self, cxr_classification_result):
        resp = client.post("/api/export/dicom_sr", data={"analysis_json": json.dumps(cxr_classification_result)})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/dicom"
        ds = pydicom.dcmread(io.BytesIO(resp.content))
        assert ds.Modality == "SR"

    def test_api_export_csv_download(self, ct_totalseg_result):
        resp = client.post("/api/export/csv", data={"analysis_json": json.dumps(ct_totalseg_result)})
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "structure_name,label_id" in resp.text

    def test_api_export_safeguard_blocks_invalid_format(self, cxr_classification_result):
        # CXR classification must reject DICOM SEG
        resp = client.post("/api/export/dicom_seg", data={"analysis_json": json.dumps(cxr_classification_result)})
        assert resp.status_code == 400
        err = resp.json()
        assert "not technically applicable" in err["detail"]

    def test_api_export_safeguard_blocks_csv_for_classification(self, cxr_classification_result):
        resp = client.post("/api/export/csv", data={"analysis_json": json.dumps(cxr_classification_result)})
        assert resp.status_code == 400
        err = resp.json()
        assert "not technically applicable" in err["detail"]
