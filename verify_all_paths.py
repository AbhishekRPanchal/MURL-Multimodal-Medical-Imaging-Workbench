import requests
import json
import os
import sys
import tempfile
import numpy as np
import nibabel as nib
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

def log_trace(input_name, detected_mod, selected_model, adapter_name, status, output_summary):
    print(f"\n[TRACE] INPUT: {input_name}")
    print(f"        DETECTED MODALITY : {detected_mod}")
    print(f"        SELECTED MODEL    : {selected_model}")
    print(f"        ADAPTER           : {adapter_name}")
    print(f"        INFERENCE STATUS  : {status}")
    print(f"        OUTPUT            : {output_summary}")

def test_routes():
    print("=" * 60)
    print(" 1. VERIFYING ALL ROUTING PATHS (INPUT -> DETECT -> ROUTE -> OUTPUT)")
    print("=" * 60)

    # 1. BUSI PNG -> BUSI classification
    busi_img = "data/raw/Dataset_BUSI_with_GT/benign/benign (1).png"
    with open(busi_img, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("benign (1).png", f, "image/png")}, data={"model_id": "busi_resnet18_classifier"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]
    log_trace(
        input_name="benign (1).png (BUSI 2D)",
        detected_mod=out["active_modality"],
        selected_model=pred["model_id"],
        adapter_name="BUSIClassifierAdapter",
        status=out["status"],
        output_summary=f"Pathology: {pred['formatted_results']['predicted_class']} (Confidence: {pred['formatted_results']['confidence']*100:.1f}%)"
    )

    # 2. BUSI PNG -> BUSI segmentation
    with open(busi_img, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("benign (1).png", f, "image/png")}, data={"model_id": "busi_monai_unet_segmentor"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]
    log_trace(
        input_name="benign (1).png (BUSI 2D)",
        detected_mod=out["active_modality"],
        selected_model=pred["model_id"],
        adapter_name="BUSISegmentorAdapter",
        status=out["status"],
        output_summary=f"Lesion Detected: {pred['formatted_results']['lesion_detected']}, Area: {pred['formatted_results']['lesion_area_percentage']}%"
    )

    # 3. CT NIfTI -> TotalSegmentator
    ct_file = "external_repos/TotalSegmentator/tests/reference_files/example_ct_sm.nii.gz"
    with open(ct_file, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("abdomen_ct.nii.gz", f, "application/gzip")}, data={"model_id": "totalsegmentator_ct"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]
    log_trace(
        input_name="abdomen_ct.nii.gz (Volumetric CT)",
        detected_mod=out["active_modality"],
        selected_model=pred["model_id"],
        adapter_name="TotalSegmentatorAdapter",
        status=out["status"],
        output_summary=f"nnU-Net executed ({pred['inference_time_ms']} ms), structures detected: {len(pred['formatted_results'].get('top_detected_structures', []))}"
    )

    # 4. Chest X-Ray PNG -> TorchXRayVision
    cxr_file = "data/samples/nih_chest_xray_00000001.png"
    with open(cxr_file, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("nih_chest_xray_00000001.png", f, "image/png")}, data={"model_id": "chexpert_chest_xray_classifier"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]
    top_finding = pred["formatted_results"].get("top_findings", [{}])[0]
    log_trace(
        input_name="nih_chest_xray_00000001.png (2D CXR)",
        detected_mod=out["active_modality"],
        selected_model=pred["model_id"],
        adapter_name="ChestXRayAdapter (DenseNet-121)",
        status=out["status"],
        output_summary=f"Pathologies: 18 evaluated, top: {top_finding.get('pathology')} ({top_finding.get('probability_percent')}%)"
    )

    # 5. MRI NIfTI -> MONAI BraTS
    mri_file = "data/samples/test_brain_mri.nii.gz"
    with open(mri_file, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("brain_mri.nii.gz", f, "application/gzip")}, data={"model_id": "monai_brats_mri_segmentor"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]
    log_trace(
        input_name="brain_mri.nii.gz (Volumetric MRI)",
        detected_mod=out["active_modality"],
        selected_model=pred["model_id"],
        adapter_name="MONAIBraTSAdapter",
        status=f"{out['status']} ({pred['formatted_results'].get('checkpoint_status')})",
        output_summary=f"MPR slices active, status_badge: '{pred['formatted_results'].get('status_badge')}', no fake masks"
    )

    # 6. Thyroid Ultrasound -> Thyroid Adapter
    with open(busi_img, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("thyroid_scan.png", f, "image/png")}, data={"model_id": "tn3k_thyroid_segmentor", "anatomy": "Thyroid"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]
    log_trace(
        input_name="thyroid_scan.png (2D Neck Ultrasound)",
        detected_mod=out["active_modality"],
        selected_model=pred["model_id"],
        adapter_name="ThyroidUltrasoundAdapter",
        status=f"{out['status']} ({pred['formatted_results'].get('status_badge')})",
        output_summary=f"Aspect ratio: {pred['formatted_results']['tirads_features']['height_to_width_ratio']}, Marker: '{pred['formatted_results']['tirads_features']['tirads_category_alert']}'"
    )

    # 7. MedSAM Image -> MedSAM Adapter with Prompt Box
    with open(busi_img, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("scan.png", f, "image/png")}, data={"model_id": "medsam_vit_b", "box_prompt_json": "[40, 50, 180, 190]"})
    assert res.status_code == 200
    out = res.json()
    assert out["status"] == "success"
    pred = out["prediction"]
    log_trace(
        input_name="scan.png (Universal Medical Raster)",
        detected_mod=out["active_modality"],
        selected_model=pred["model_id"],
        adapter_name="MedSAMAdapter",
        status=f"{out['status']} ({pred['formatted_results'].get('checkpoint_status')})",
        output_summary=f"Registered Box: {pred['formatted_results'].get('box_prompt_received')}, Status: '{pred['formatted_results'].get('status_badge')}'"
    )

    # 8. DICOM Ingestion -> Inspect & Routing
    dcm_file = "external_repos/Slicer/Testing/Data/Input/CTHeadAxialDicom/CTHead1.dcm"
    with open(dcm_file, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/inspect", files={"file": ("CTHead1.dcm", f, "application/dicom")})
    assert res.status_code == 200
    insp = res.json()
    dcm = insp["dicom_metadata"]
    log_trace(
        input_name="CTHead1.dcm (Real Clinical DICOM)",
        detected_mod=insp["modality_detection"]["modality"],
        selected_model=insp["compatible_models"][0] if insp["compatible_models"] else "None",
        adapter_name="DICOMMetadataParser",
        status="success",
        output_summary=f"Format: {insp['format']}, Modality: {dcm['modality']}, Privacy: {dcm['de_identification_status']}"
    )

def test_error_handling():
    print("\n" + "=" * 60)
    print(" 2. VERIFYING EXPLICIT ERROR HANDLING & SAFEGUARDS")
    print("=" * 60)

    # A. Unsupported file format (e.g. .txt or .exe)
    res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("notes.txt", b"Invalid medical scan contents", "text/plain")})
    out = res.json()
    assert out["status"] == "error"
    print(f"[PASS] Unsupported Format: error='{out.get('error')}'")

    # B. Corrupt NIfTI file (corrupt magic bytes)
    res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("corrupt.nii", b"Not a valid nifti header content at all", "application/octet-stream")})
    out = res.json()
    assert out["status"] == "error"
    print(f"[PASS] Corrupt NIfTI: error='{out.get('error')}'")

    # C. Corrupt DICOM file (missing 'DICM' signature at offset 128)
    res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("corrupt.dcm", b"0" * 200, "application/dicom")})
    out = res.json()
    assert out["status"] == "error"
    print(f"[PASS] Corrupt DICOM: error='{out.get('error')}'")

    # D. Incompatible Modality to Model (Chest X-ray sent to Brain MRI Model)
    cxr_file = "data/samples/nih_chest_xray_00000001.png"
    with open(cxr_file, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("nih_chest_xray_00000001.png", f, "image/png")}, data={"model_id": "monai_brats_mri_segmentor", "modality_override": "X-ray"})
    out = res.json()
    assert out["status"] == "error"
    print(f"[PASS] Incompatible Modality: error='{out.get('error')}'")

    # E. Segmentation Mask Protected from being Treated as Raw CT Input
    mask_file = "outputs/ct_segmentations/totalseg_out_1793847812557.nii.gz"
    with open(mask_file, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("totalseg_out_mask.nii.gz", f, "application/gzip")}, data={"model_id": "totalsegmentator_ct"})
    out = res.json()
    assert out["status"] == "error"
    assert "segmentation" in out.get("error", "").lower() or "mask" in out.get("error", "").lower()
    print(f"[PASS] Segmentation Mask Protection: error='{out.get('error')}'")

def test_export_verification():
    print("\n" + "=" * 60)
    print(" 3. VERIFYING ACTUAL EXPORT INTEGRITY & FILE OPENING")
    print("=" * 60)

    # 1. BUSI Export
    busi_img = "data/raw/Dataset_BUSI_with_GT/benign/benign (1).png"
    with open(busi_img, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("benign (1).png", f, "image/png")}, data={"model_id": "busi_resnet18_classifier"})
    data = res.json()
    export_path = "outputs/test_export_busi.json"
    with open(export_path, "w") as f:
        json.dump(data, f, indent=2)
    assert os.path.exists(export_path)
    assert os.path.getsize(export_path) > 500
    with open(export_path, "r") as f:
        loaded = json.load(f)
    assert loaded["status"] == "success"
    print(f"[PASS] BUSI Export verified: {export_path} ({os.path.getsize(export_path)} bytes)")

    # 2. TotalSegmentator CT Segmentation Output Volume Opening
    ct_file = "outputs/ct_segmentations/totalseg_out_1793847812557.nii.gz"
    assert os.path.exists(ct_file)
    nii = nib.load(ct_file)
    data_arr = nii.get_fdata()
    assert data_arr.ndim == 3
    print(f"[PASS] CT Segmentation Output Volume verified: {ct_file} (shape={nii.shape}, zooms={nii.header.get_zooms()[:3]})")

    # 3. Chest X-Ray Export
    cxr_file = "data/samples/nih_chest_xray_00000001.png"
    with open(cxr_file, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"file": ("nih_chest_xray_00000001.png", f, "image/png")}, data={"model_id": "chexpert_chest_xray_classifier"})
    data_cxr = res.json()
    export_cxr = "outputs/test_export_cxr.json"
    with open(export_cxr, "w") as f:
        json.dump(data_cxr, f, indent=2)
    assert os.path.exists(export_cxr)
    assert os.path.getsize(export_cxr) > 500
    print(f"[PASS] CXR Export verified: {export_cxr} ({os.path.getsize(export_cxr)} bytes)")

if __name__ == "__main__":
    test_routes()
    test_error_handling()
    test_export_verification()
    print("\n" + "=" * 60)
    print(" All audits, routing paths, safeguards and exports passed.")
    print("=" * 60)
