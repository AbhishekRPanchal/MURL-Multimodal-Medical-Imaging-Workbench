import requests
import json
import os
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    data = res.json()
    assert data["status"] == "healthy"
    print(f"[PASS] Health: {data['status']}, device: {data['device']}, version: {data['version']}")

def test_models():
    res = requests.get(f"{BASE_URL}/api/models")
    assert res.status_code == 200, f"Models check failed: {res.text}"
    models = res.json()
    model_ids = [m["model_id"] for m in models]
    print(f"[PASS] Loaded {len(models)} models: {model_ids}")
    return model_ids

def test_busi_resnet18():
    busi_img = "data/raw/Dataset_BUSI_with_GT/benign/benign (1).png"
    with open(busi_img, "rb") as f:
        files = {"file": ("benign (1).png", f, "image/png")}
        data = {"model_id": "busi_resnet18_classifier"}
        res = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    out = res.json()
    assert out["status"] == "success", f"Inference unsuccessful: {out.get('error')}"
    pred = out["prediction"]
    res_fmt = pred["formatted_results"]
    assert res_fmt["predicted_class"] == "benign"
    print(f"[PASS] busi_resnet18_classifier: {res_fmt['predicted_class']} (confidence: {res_fmt['confidence']:.4f}, time: {pred['inference_time_ms']}ms)")

def test_busi_unet():
    busi_img = "data/raw/Dataset_BUSI_with_GT/benign/benign (1).png"
    with open(busi_img, "rb") as f:
        files = {"file": ("benign (1).png", f, "image/png")}
        data = {"model_id": "busi_monai_unet_segmentor"}
        res = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    out = res.json()
    assert out["status"] == "success", f"Inference unsuccessful: {out.get('error')}"
    pred = out["prediction"]
    res_fmt = pred["formatted_results"]
    area = res_fmt.get("lesion_area_percentage", res_fmt.get("lesion_area_fraction", 0.0))
    print(f"[PASS] busi_monai_unet_segmentor: lesion_area_percentage = {area}%, time: {pred['inference_time_ms']}ms")

def test_totalsegmentator_ct():
    ct_file = "outputs/ct_segmentations/totalseg_out_1793847812557.nii.gz"
    with open(ct_file, "rb") as f:
        files = {"file": ("test_ct.nii.gz", f, "application/gzip")}
        data = {"model_id": "totalsegmentator_ct"}
        res = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    out = res.json()
    assert out["status"] == "success", f"Inference unsuccessful: {out.get('error')}"
    pred = out["prediction"]
    res_fmt = pred["formatted_results"]
    structures = res_fmt.get("structures_detected", [])
    print(f"[PASS] totalsegmentator_ct: detected {len(structures)} structures, time: {pred['inference_time_ms']}ms")

def test_chexpert_xray():
    xray_file = "data/samples/nih_chest_xray_00000001.png"
    with open(xray_file, "rb") as f:
        files = {"file": ("nih_chest_xray_00000001.png", f, "image/png")}
        data = {"model_id": "chexpert_chest_xray_classifier"}
        res = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    out = res.json()
    assert out["status"] == "success", f"Inference unsuccessful: {out.get('error')}"
    pred = out["prediction"]
    res_fmt = pred["formatted_results"]
    top_findings = res_fmt.get("top_findings", [])
    pred_cls = top_findings[0]["pathology"] if top_findings else res_fmt.get("predicted_class", "unknown")
    findings = res_fmt.get("all_findings", res_fmt.get("pathology_probabilities", []))
    print(f"[PASS] chexpert_chest_xray_classifier: top pathology: {pred_cls}, evaluated count: {len(findings)}, time: {pred['inference_time_ms']}ms")

def test_auto_route_busi():
    busi_img = "data/raw/Dataset_BUSI_with_GT/benign/benign (1).png"
    with open(busi_img, "rb") as f:
        files = {"file": ("benign (1).png", f, "image/png")}
        data = {"model_id": "", "modality_override": "Ultrasound"}
        res = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    out = res.json()
    assert out["status"] == "success", f"Auto-route failed: {out.get('error')}"
    routed_model = out["prediction"]["model_id"]
    print(f"[PASS] Auto-route with Ultrasound override routed to: {routed_model}")

def test_adapter_ready_models():
    busi_img = "data/raw/Dataset_BUSI_with_GT/benign/benign (1).png"
    # TN3K
    with open(busi_img, "rb") as f:
        files = {"file": ("benign (1).png", f, "image/png")}
        data = {"model_id": "tn3k_thyroid_segmentor"}
        res = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
    out = res.json()
    assert out["status"] == "success", f"TN3K analysis failed: {out.get('error')}"
    print(f"[PASS] tn3k_thyroid_segmentor: status = {out['status']}, adapter = {out['prediction']['model_id']}")

    # MedSAM
    with open(busi_img, "rb") as f:
        files = {"file": ("benign (1).png", f, "image/png")}
        data = {"model_id": "medsam_vit_b", "box_prompt_json": "[50, 50, 200, 200]"}
        res = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data)
    out = res.json()
    assert out["status"] == "success", f"MedSAM analysis failed: {out.get('error')}"
    print(f"[PASS] medsam_vit_b: status = {out['status']}, adapter = {out['prediction']['model_id']}")

if __name__ == "__main__":
    print("=== STARTING COMPLETE E2E API AUDIT ===")
    test_health()
    test_models()
    test_busi_resnet18()
    test_busi_unet()
    test_totalsegmentator_ct()
    test_chexpert_xray()
    test_auto_route_busi()
    test_adapter_ready_models()
    print("=== All target model pipelines and adapters verified operational ===")
