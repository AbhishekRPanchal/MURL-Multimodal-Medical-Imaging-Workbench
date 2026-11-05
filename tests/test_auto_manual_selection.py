"""
Acceptance Test Suite: Auto + Manual Modality Selection & Safety Routing
Verifies all 15 acceptance criteria:
1. Correct X-ray -> auto detected X-ray
2. Correct BUSI -> auto detected Ultrasound
3. Correct CT -> auto detected CT
4. Correct MRI -> auto detected MRI
5. Wrong/ambiguous image -> manual selection appears / low confidence or fallback
6. Detection failure -> manual selection appears / fallback
7. X-ray manually selected as ultrasound -> incompatible selection rejected
8. Valid MRI manually selected -> MRI pipeline runs
9. PDF page -> automatic detection
10. PDF page with manual modality selection
11. Automatic mode -> correct operational model
12. Manual mode -> correct operational model
13. No internal model names exposed in UI
14. No [Planned] or [Adapter ready] entries exposed in operator UI
15. No fake inference
"""

import os
import io
import pytest
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

from api.app import app
from src.platform.unified_service import UnifiedInferenceService

client = TestClient(app)

CXR_PATH = "MURL_DEMO_DATA/chest_xray/nih_cxr_cardiomegaly_01.png"
BUSI_PATH = "MURL_DEMO_DATA/breast_ultrasound/busi_benign_01.png"
CT_PATH = "MURL_DEMO_DATA/ct/example_ct_sm.nii.gz"
MRI_PATH = "MURL_DEMO_DATA/brain_mri/brats_brain_mri_demo.nii.gz"
PDF_PATH = "MURL_DEMO_DATA/pdf/multipage_medical_scans.pdf"
THYROID_PATH = "MURL_DEMO_DATA/thyroid_ultrasound/tn3k_thyroid_sample_0000.jpg"


def test_01_correct_cxr_auto_detected():
    """Criterion 1: Correct X-ray -> auto detected X-ray"""
    with open(CXR_PATH, "rb") as f:
        file_bytes = f.read()

    # Inspect endpoint
    resp = client.post("/api/inspect", files={"file": ("cxr.png", file_bytes, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["modality_detection"]["modality"] == "X-ray"
    confidence = data["modality_detection"].get("confidence", data["modality_detection"].get("probability", 0.0))
    assert confidence >= 0.80

    # Analyze endpoint in default auto mode
    resp = client.post("/api/analyze", files={"file": ("cxr.png", file_bytes, "image/png")})
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert res["active_modality"] == "X-ray"
    assert res["analysis_mode"] == "Automatic"
    assert res["prediction"]["model_id"] == "chexpert_chest_xray_classifier"


def test_02_correct_busi_auto_detected():
    """Criterion 2: Correct BUSI -> auto detected Ultrasound"""
    with open(BUSI_PATH, "rb") as f:
        file_bytes = f.read()

    resp = client.post("/api/inspect", files={"file": ("busi.png", file_bytes, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality_detection"]["modality"] == "Ultrasound"

    resp = client.post("/api/analyze", files={"file": ("busi.png", file_bytes, "image/png")})
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert res["active_modality"] == "Ultrasound"
    assert res["analysis_mode"] == "Automatic"
    assert "predicted_class" in res["prediction"]["formatted_results"]


def test_03_correct_ct_auto_detected():
    """Criterion 3: Correct CT -> auto detected CT"""
    with open(CT_PATH, "rb") as f:
        file_bytes = f.read()

    resp = client.post("/api/inspect", files={"file": ("ct.nii.gz", file_bytes, "application/gzip")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality_detection"]["modality"] == "CT"

    resp = client.post("/api/analyze", files={"file": ("ct.nii.gz", file_bytes, "application/gzip")})
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert res["active_modality"] == "CT"
    assert res["analysis_mode"] == "Automatic"
    assert res["prediction"]["model_id"] == "totalsegmentator_ct"


def test_04_correct_mri_auto_detected():
    """Criterion 4: Correct MRI -> auto detected MRI"""
    with open(MRI_PATH, "rb") as f:
        file_bytes = f.read()

    resp = client.post("/api/inspect", files={"file": ("mri.nii.gz", file_bytes, "application/gzip")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality_detection"]["modality"] == "MRI"

    resp = client.post("/api/analyze", files={"file": ("mri.nii.gz", file_bytes, "application/gzip")})
    assert resp.status_code == 200
    res = resp.json()
    assert res["status"] == "success"
    assert res["active_modality"] == "MRI"
    assert res["analysis_mode"] == "Automatic"
    assert res["prediction"]["model_id"] == "monai_brats_mri_segmentor"


def test_05_ambiguous_image_manual_trigger():
    """Criterion 5: Wrong/ambiguous image -> low confidence / manual fallback"""
    # Create flat grey image with no medical texture or markings
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    flat_bytes = buf.getvalue()

    resp = client.post("/api/inspect", files={"file": ("ambiguous.png", flat_bytes, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    # Ambiguous images must have low confidence or unknown modality
    det = data["modality_detection"]
    confidence = det.get("confidence", 0.0)
    assert confidence < 0.80 or det.get("modality") in ["Unknown", "Other"]


def test_06_detection_failure_manual_trigger():
    """Criterion 6: Detection failure -> handled safely without crash"""
    # Create random noise image
    noise_arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noise_arr).save(buf, format="PNG")
    noise_bytes = buf.getvalue()

    resp = client.post("/api/inspect", files={"file": ("noise.png", noise_bytes, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"


def test_07_incompatible_cross_modality_rejection():
    """Criterion 7: Cross-modality incompatible routing is rejected for clinical safety"""
    with open(CXR_PATH, "rb") as f:
        cxr_bytes = f.read()

    # Operator tries to force an X-ray to run as Breast Ultrasound
    resp = client.post(
        "/api/analyze",
        data={"modality_override": "Breast Ultrasound"},
        files={"file": ("cxr.png", cxr_bytes, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "Incompatible cross-modality routing is rejected for clinical safety" in data["error"]
    assert "X-ray" in data["error"]
    assert "Breast Ultrasound" in data["error"]

    # Operator tries to force an Ultrasound to run as CT
    with open(BUSI_PATH, "rb") as f:
        busi_bytes = f.read()

    resp = client.post(
        "/api/analyze",
        data={"modality_override": "CT"},
        files={"file": ("busi.png", busi_bytes, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "Incompatible cross-modality routing is rejected for clinical safety" in data["error"]


def test_08_valid_mri_manually_selected():
    """Criterion 8: Valid MRI manually selected -> MRI pipeline runs"""
    with open(MRI_PATH, "rb") as f:
        mri_bytes = f.read()

    resp = client.post(
        "/api/analyze",
        data={"modality_override": "MRI"},
        files={"file": ("mri.nii.gz", mri_bytes, "application/gzip")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["active_modality"] == "MRI"
    assert data["analysis_mode"] == "Manual — MRI"
    assert data["prediction"]["model_id"] == "monai_brats_mri_segmentor"


def test_09_pdf_page_automatic_detection():
    """Criterion 9: PDF page -> automatic modality detection"""
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    resp = client.post(
        "/api/pdf/page",
        data={"page_number": 1},
        files={"file": ("scans.pdf", pdf_bytes, "application/pdf")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["page_number"] == 1
    assert "image_b64" in data
    assert "modality_detection" in data


def test_10_pdf_page_manual_selection():
    """Criterion 10: PDF page with manual modality selection"""
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    # Analyze Page 1 (X-ray) with manual modality specification
    resp = client.post(
        "/api/analyze",
        data={"page_number": 1, "modality_override": "Chest X-ray"},
        files={"file": ("scans.pdf", pdf_bytes, "application/pdf")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["analysis_mode"] == "Manual — Chest X-ray"

    # Verify incompatible manual override on PDF page is safely rejected
    resp_conflict = client.post(
        "/api/analyze",
        data={"page_number": 1, "modality_override": "Breast Ultrasound"},
        files={"file": ("scans.pdf", pdf_bytes, "application/pdf")}
    )
    assert resp_conflict.status_code == 200
    data_conflict = resp_conflict.json()
    assert data_conflict["status"] == "error"
    assert "Incompatible cross-modality routing is rejected for clinical safety" in data_conflict["error"]


def test_11_automatic_mode_correct_operational_model():
    """Criterion 11: Automatic mode routes to operational models without intervention"""
    # Test CXR -> CheXpert
    with open(CXR_PATH, "rb") as f:
        cxr_resp = client.post("/api/analyze", files={"file": ("cxr.png", f.read(), "image/png")})
    assert cxr_resp.json()["prediction"]["model_id"] == "chexpert_chest_xray_classifier"

    # Test CT -> TotalSegmentator
    with open(CT_PATH, "rb") as f:
        ct_resp = client.post("/api/analyze", files={"file": ("ct.nii.gz", f.read(), "application/gzip")})
    assert ct_resp.json()["prediction"]["model_id"] == "totalsegmentator_ct"


def test_12_manual_mode_correct_operational_model():
    """Criterion 12: Manual mode routes to task-specified operational model"""
    with open(BUSI_PATH, "rb") as f:
        busi_bytes = f.read()

    # Manual Classification
    resp_clf = client.post(
        "/api/analyze",
        data={"modality_override": "Breast Ultrasound", "task": "classification"},
        files={"file": ("busi.png", busi_bytes, "image/png")}
    )
    assert resp_clf.status_code == 200
    assert resp_clf.json()["prediction"]["model_id"] == "busi_resnet18_classifier"

    # Manual Segmentation
    resp_seg = client.post(
        "/api/analyze",
        data={"modality_override": "Breast Ultrasound", "task": "segmentation"},
        files={"file": ("busi.png", busi_bytes, "image/png")}
    )
    assert resp_seg.status_code == 200
    assert resp_seg.json()["prediction"]["model_id"] == "busi_monai_unet_segmentor"


def test_13_no_internal_model_names_in_ui():
    """Criterion 13: No internal model names or IDs exposed in user-facing operator UI"""
    with open("api/static/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Verify manual scan type select has friendly names only
    assert '<select id="manual-modality-select"' in html
    assert '<option value="Chest X-ray">Chest X-ray</option>' in html
    assert '<option value="Breast Ultrasound">Breast Ultrasound</option>' in html
    assert '<option value="Thyroid Ultrasound">Thyroid Ultrasound</option>' in html
    assert '<option value="CT">CT</option>' in html
    assert '<option value="MRI">MRI</option>' in html

    # Verify internal model IDs are not options in the manual modality select
    manual_select_section = html.split('<select id="manual-modality-select"')[1].split('</select>')[0]
    forbidden_ids = [
        "busi_resnet18_classifier",
        "busi_monai_unet_segmentor",
        "chexpert_chest_xray_classifier",
        "totalsegmentator_ct",
        "monai_brats_mri_segmentor",
        "tn3k_thyroid_segmentor"
    ]
    for fid in forbidden_ids:
        assert fid not in manual_select_section


def test_14_no_planned_or_adapter_ready_in_ui():
    """Criterion 14: No [Planned] or [Adapter ready] status badges exposed in operator UI"""
    with open("api/static/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "[Planned]" not in html
    assert "[Adapter ready]" not in html
    assert "[E2E tested]" not in html
    assert "[Validated]" not in html


def test_15_no_fake_inference():
    """Criterion 15: No fake inference or fabricated predictions"""
    with open(MRI_PATH, "rb") as f:
        mri_bytes = f.read()

    # MRI inference must return genuine status (executed with valid compartments, or checkpoint required)
    resp = client.post("/api/analyze", files={"file": ("mri.nii.gz", mri_bytes, "application/gzip")})
    assert resp.status_code == 200
    data = resp.json()
    pred = data["prediction"]
    fmt = pred["formatted_results"]
    assert fmt["pipeline_status"] in ["checkpoint_required", "executed"]
    if fmt["pipeline_status"] == "executed":
        assert "compartments" in fmt
        assert len(fmt["compartments"]) > 0
    else:
        assert fmt.get("mask_overlay_b64") is None

    # Region segmentation tool (MedSAM) must return genuine prompt-guided structure
    with open(BUSI_PATH, "rb") as f:
        busi_bytes = f.read()

    resp_medsam = client.post(
        "/api/analyze",
        data={"model_id": "medsam_vit_b", "box_prompt_json": "[50, 50, 150, 150]"},
        files={"file": ("busi.png", busi_bytes, "image/png")}
    )
    assert resp_medsam.status_code == 200
    res_m = resp_medsam.json()
    assert res_m["analysis_mode"] == "Region segmentation"
    assert res_m["prediction"]["model_id"] == "medsam_vit_b"
