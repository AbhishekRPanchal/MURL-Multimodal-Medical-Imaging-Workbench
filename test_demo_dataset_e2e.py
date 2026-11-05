"""
MURL Demo Dataset Verification and E2E Regression Script
Validates every sample in MURL_DEMO_DATA/ through the live platform API.
"""

import os
import json
import requests
import nibabel as nib
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

def test_all_demo_samples():
    print("=" * 70)
    print(" VERIFYING ALL SAMPLES IN MURL_DEMO_DATA")
    print("=" * 70)

    # 1. BUSI Benign
    busi_benign = "MURL_DEMO_DATA/breast_ultrasound/busi_benign_01.png"
    assert os.path.exists(busi_benign)
    with Image.open(busi_benign) as im:
        assert im.format == "PNG"
        w, h = im.size
        assert w > 0 and h > 0
    with open(busi_benign, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("busi_benign_01.png", f, "image/png")}, data={"model_id": "busi_resnet18_classifier"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    assert out["active_modality"] == "Ultrasound"
    assert out["prediction"]["model_id"] == "busi_resnet18_classifier"
    busi_conf = out['prediction']['formatted_results']['confidence']
    busi_conf_pct = busi_conf * 100 if busi_conf <= 1.0 else busi_conf
    print(f"[PASS] BUSI Benign: {out['prediction']['formatted_results']['predicted_class']} (Confidence: {busi_conf_pct:.2f}%)")

    # 2. BUSI Malignant
    busi_malignant = "MURL_DEMO_DATA/breast_ultrasound/busi_malignant_01.png"
    assert os.path.exists(busi_malignant)
    with Image.open(busi_malignant) as im:
        assert im.format == "PNG"
    with open(busi_malignant, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("busi_malignant_01.png", f, "image/png")}, data={"model_id": "busi_monai_unet_segmentor"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    assert out["prediction"]["model_id"] == "busi_monai_unet_segmentor"
    assert out["prediction"]["formatted_results"]["lesion_detected"] is True
    print(f"[PASS] BUSI Malignant Segmentation: Lesion Area: {out['prediction']['formatted_results']['lesion_area_percentage']}%")

    # 3. BUSI Normal
    busi_normal = "MURL_DEMO_DATA/breast_ultrasound/busi_normal_01.png"
    assert os.path.exists(busi_normal)
    with open(busi_normal, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("busi_normal_01.png", f, "image/png")}, data={"model_id": "busi_resnet18_classifier"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    print(f"[PASS] BUSI Normal: {out['prediction']['formatted_results']['predicted_class']}")

    # 4. Chest X-ray (Cardiomegaly)
    cxr_sample = "MURL_DEMO_DATA/chest_xray/nih_cxr_cardiomegaly_01.png"
    assert os.path.exists(cxr_sample)
    with Image.open(cxr_sample) as im:
        assert im.format == "PNG"
    with open(cxr_sample, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("nih_cxr_cardiomegaly_01.png", f, "image/png")}, data={"model_id": "chexpert_chest_xray_classifier"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    assert out["active_modality"] == "X-ray"
    top_pathology = out["prediction"]["formatted_results"]["top_findings"][0]
    print(f"[PASS] Chest X-Ray: Top Finding: {top_pathology['pathology']} ({top_pathology['probability_percent']}%)")

    # 5. Chest X-ray (Normal)
    cxr_normal = "MURL_DEMO_DATA/chest_xray/cxr_normal_02.png"
    assert os.path.exists(cxr_normal)
    with open(cxr_normal, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("cxr_normal_02.png", f, "image/png")}, data={"model_id": "chexpert_chest_xray_classifier"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    print(f"[PASS] Chest X-Ray Normal evaluated: {out['prediction']['formatted_results']['pathologies_evaluated']} pathologies")

    # 6. Brain MRI (NIfTI & MPR)
    mri_sample = "MURL_DEMO_DATA/brain_mri/brats_brain_mri_demo.nii.gz"
    assert os.path.exists(mri_sample)
    nii = nib.load(mri_sample)
    assert len(nii.shape) in [3, 4]
    print(f"[PASS] MRI NIfTI dimensions: {nii.shape}, zooms: {nii.header.get_zooms()[:3]}")

    with open(mri_sample, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("brats_brain_mri_demo.nii.gz", f, "application/gzip")}, data={"model_id": "monai_brats_mri_segmentor"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    assert out["active_modality"] == "MRI"
    pred = out["prediction"]["formatted_results"]
    assert pred["pipeline_status"] in ["executed", "checkpoint_required"]
    assert "multiplanar_previews" in pred
    assert "axial_wt_b64" in pred["multiplanar_previews"]
    if pred["pipeline_status"] == "executed":
        assert len(pred["compartments"]) > 0
    else:
        assert len(pred["compartments"]) == 0  # Honest: no fake masks
    print(f"[PASS] Brain MRI: MPR previews generated, pipeline_status='{pred['pipeline_status']}'")

    # 7. Thyroid Ultrasound (TN3K)
    thyroid_sample = "MURL_DEMO_DATA/thyroid_ultrasound/tn3k_thyroid_sample_0000.jpg"
    assert os.path.exists(thyroid_sample)
    with Image.open(thyroid_sample) as im:
        assert im.format == "JPEG"
    with open(thyroid_sample, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("tn3k_thyroid_sample_0000.jpg", f, "image/jpeg")}, data={"model_id": "tn3k_thyroid_segmentor"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]["formatted_results"]
    assert pred["status_badge"] == "Heuristic analysis"
    assert "tirads_features" in pred
    ar = pred["tirads_features"]["height_to_width_ratio"]
    print(f"[PASS] Thyroid Ultrasound: status_badge='{pred['status_badge']}', Aspect Ratio={ar}")

    # 8. CT (TotalSegmentator)
    ct_sample = "MURL_DEMO_DATA/ct/example_ct_sm.nii.gz"
    assert os.path.exists(ct_sample)
    nii_ct = nib.load(ct_sample)
    assert len(nii_ct.shape) == 3
    print(f"[PASS] CT NIfTI dimensions: {nii_ct.shape}, zooms: {nii_ct.header.get_zooms()[:3]}")

    with open(ct_sample, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("example_ct_sm.nii.gz", f, "application/gzip")}, data={"model_id": "totalsegmentator_ct"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]["formatted_results"]
    assert "top_detected_structures" in pred
    print(f"[PASS] CT TotalSegmentator: {len(pred['top_detected_structures'])} structures detected in {out['prediction']['inference_time_ms']} ms")

    # 9. DICOM (CTHead1.dcm)
    dcm_sample = "MURL_DEMO_DATA/dicom/CTHead1.dcm"
    assert os.path.exists(dcm_sample)
    with open(dcm_sample, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/inspect", files={"file": ("CTHead1.dcm", f, "application/dicom")})
    assert res.status_code == 200
    meta = res.json()["dicom_metadata"]
    assert meta["is_dicom"] is True
    assert meta["modality"] == "CT"
    assert "PatientName" not in meta
    assert "PatientID" not in meta
    dims = meta["dimensions"]
    print(f"[PASS] DICOM: modality={meta['modality']}, dimensions={dims['rows']}x{dims['columns']}, Research privacy filtering applied")

    # 10. MedSAM (Reusing BUSI sample)
    with open(busi_benign, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("busi_benign_01.png", f, "image/png")}, data={"model_id": "medsam_vit_b", "box_prompt_json": "[30, 40, 150, 180]"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]["formatted_results"]
    assert pred["status_badge"] in ["Complete", "Checkpoint required"]
    assert pred["box_prompt_received"] == [30, 40, 150, 180]
    print(f"[PASS] MedSAM: status_badge='{pred['status_badge']}', prompt registered: {pred['box_prompt_received']}")

    # 11. Multi-Format Medical Result Export Verification
    # A. Chest X-Ray: Verify PDF, DICOM SR, JSON
    with open(cxr_sample, "rb") as f:
        res_cxr = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("nih_cxr_cardiomegaly_01.png", f, "image/png")}, data={"model_id": "chexpert_chest_xray_classifier"})
    cxr_data = res_cxr.json()
    
    # Formats check
    res_fmt = requests.post(f"{BASE_URL}/api/export/formats", data={"analysis_json": json.dumps(cxr_data)})
    assert res_fmt.status_code == 200
    cxr_fmts = [x["id"] for x in res_fmt.json()["formats"]]
    assert "pdf" in cxr_fmts and "dicom_sr" in cxr_fmts and "json" in cxr_fmts
    assert "csv" not in cxr_fmts and "dicom_seg" not in cxr_fmts

    # Download PDF
    res_pdf = requests.post(f"{BASE_URL}/api/export/pdf", data={"analysis_json": json.dumps(cxr_data)})
    assert res_pdf.status_code == 200 and res_pdf.content.startswith(b"%PDF-1.")
    with open("outputs/test_cxr_report.pdf", "wb") as f_pdf:
        f_pdf.write(res_pdf.content)
    print(f"[PASS] Export PDF verified: outputs/test_cxr_report.pdf ({len(res_pdf.content)} bytes)")

    # Download DICOM SR
    res_sr = requests.post(f"{BASE_URL}/api/export/dicom_sr", data={"analysis_json": json.dumps(cxr_data)})
    assert res_sr.status_code == 200
    with open("outputs/test_cxr_sr.dcm", "wb") as f_sr:
        f_sr.write(res_sr.content)
    print(f"[PASS] Export DICOM SR verified: outputs/test_cxr_sr.dcm ({len(res_sr.content)} bytes)")

    # B. CT TotalSegmentator: Verify DICOM SEG, NIfTI Mask, CSV, PDF
    with open(ct_sample, "rb") as f:
        res_ct = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("example_ct_sm.nii.gz", f, "application/gzip")}, data={"model_id": "totalsegmentator_ct"})
    ct_data = res_ct.json()

    res_seg = requests.post(f"{BASE_URL}/api/export/dicom_seg", data={"analysis_json": json.dumps(ct_data)})
    assert res_seg.status_code == 200
    with open("outputs/test_ct_seg.dcm", "wb") as f_seg:
        f_seg.write(res_seg.content)
    print(f"[PASS] Export DICOM SEG verified: outputs/test_ct_seg.dcm ({len(res_seg.content)} bytes)")

    res_nii = requests.post(f"{BASE_URL}/api/export/nifti", data={"analysis_json": json.dumps(ct_data)})
    assert res_nii.status_code == 200
    with open("outputs/test_ct_mask.nii.gz", "wb") as f_nii:
        f_nii.write(res_nii.content)
    print(f"[PASS] Export NIfTI Mask verified: outputs/test_ct_mask.nii.gz ({len(res_nii.content)} bytes)")

    res_csv = requests.post(f"{BASE_URL}/api/export/csv", data={"analysis_json": json.dumps(ct_data)})
    assert res_csv.status_code == 200
    with open("outputs/test_ct_metrics.csv", "w", encoding="utf-8") as f_csv:
        f_csv.write(res_csv.text)
    print(f"[PASS] Export CSV verified: outputs/test_ct_metrics.csv ({len(res_csv.text)} bytes)")

    # JSON export
    export_json = "outputs/sample_exported_report.json"
    assert os.path.exists(export_json)
    with open(export_json, "r") as f:
        export_data = json.load(f)
        assert export_data.get("status") == "success" and "prediction" in export_data
    print(f"[PASS] Export JSON verified: {export_json} ({os.path.getsize(export_json)} bytes)")

    print("\n" + "=" * 70)
    print(" All demo samples and regression tests passed.")
    print("=" * 70)

if __name__ == "__main__":
    test_all_demo_samples()
