"""
Task 5: Independent API Verification Script
Tests FastAPI microservice using TestClient against:
1. Health endpoint (/health)
2. Valid ultrasound image (/predict/full, /predict/classify, /predict/segment)
3. Invalid non-image file (text upload)
4. Corrupted image file
5. Large image canvas (3000x3000px)
6. Checkpoint loading verification
Logs all HTTP statuses and response payloads to outputs/api_test_report.json.
"""

import os
import sys
import io
import json
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.app import app, get_predictor

def run_api_verification():
    print("=== FASTAPI VERIFICATION SUITE ===")
    client = TestClient(app)
    results = {}

    # Test 1: Health Endpoint
    print("\n[Test 1] Testing GET /health ...")
    t0 = time.time()
    resp_health = client.get("/health")
    latency_health = time.time() - t0
    print(f"Status: {resp_health.status_code} | Latency: {latency_health:.4f}s")
    print(f"Response: {resp_health.json()}")
    assert resp_health.status_code == 200, "Health check failed!"
    results["health_endpoint"] = {
        "status_code": resp_health.status_code,
        "latency_sec": round(latency_health, 4),
        "response": resp_health.json()
    }

    # Test 2: Valid Ultrasound Image (/predict/full)
    sample_path = "data/raw/Dataset_BUSI_with_GT/malignant/malignant (1).png"
    print(f"\n[Test 2] Testing POST /predict/full with valid image ({sample_path}) ...")
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    t0 = time.time()
    resp_full = client.post(
        "/predict/full",
        files={"file": ("malignant (1).png", img_bytes, "image/png")}
    )
    latency_full = time.time() - t0
    print(f"Status: {resp_full.status_code} | Latency: {latency_full:.4f}s")
    data_full = resp_full.json()
    assert resp_full.status_code == 200, "Full prediction failed!"
    print(f"Predicted Class: {data_full['classification']['predicted_class']} (Confidence: {data_full['classification']['confidence']:.4f})")
    print(f"Lesion Detected: {data_full['segmentation']['lesion_detected']} (Area: {data_full['segmentation']['lesion_area_percentage']}%)")
    print(f"Bounding Boxes: {data_full['segmentation'].get('derived_bounding_boxes')}")
    results["predict_full_valid"] = {
        "status_code": resp_full.status_code,
        "latency_sec": round(latency_full, 4),
        "response": data_full
    }

    # Test 3: Valid Ultrasound Image (/predict/classify)
    print("\n[Test 3] Testing POST /predict/classify ...")
    t0 = time.time()
    resp_classify = client.post(
        "/predict/classify",
        files={"file": ("malignant (1).png", img_bytes, "image/png")}
    )
    latency_classify = time.time() - t0
    print(f"Status: {resp_classify.status_code} | Latency: {latency_classify:.4f}s")
    print(f"Classification Response: {resp_classify.json()}")
    assert resp_classify.status_code == 200, "Classify endpoint failed!"
    results["predict_classify"] = {
        "status_code": resp_classify.status_code,
        "latency_sec": round(latency_classify, 4),
        "response": resp_classify.json()
    }

    # Test 4: Valid Ultrasound Image (/predict/segment)
    print("\n[Test 4] Testing POST /predict/segment ...")
    t0 = time.time()
    resp_segment = client.post(
        "/predict/segment",
        files={"file": ("malignant (1).png", img_bytes, "image/png")}
    )
    latency_segment = time.time() - t0
    print(f"Status: {resp_segment.status_code} | Latency: {latency_segment:.4f}s")
    data_seg = resp_segment.json()
    assert resp_segment.status_code == 200, "Segment endpoint failed!"
    results["predict_segment"] = {
        "status_code": resp_segment.status_code,
        "latency_sec": round(latency_segment, 4),
        "response": data_seg
    }

    # Test 5: Invalid File Upload (Plain Text)
    print("\n[Test 5] Testing invalid non-image upload ...")
    text_content = b"This is a text file, not an ultrasound image."
    resp_invalid_type = client.post(
        "/predict/full",
        files={"file": ("notes.txt", text_content, "text/plain")}
    )
    print(f"Status: {resp_invalid_type.status_code} | Response: {resp_invalid_type.json()}")
    assert resp_invalid_type.status_code == 400, "Expected HTTP 400 for text upload!"
    results["invalid_file_type"] = {
        "status_code": resp_invalid_type.status_code,
        "response": resp_invalid_type.json()
    }

    # Test 6: Corrupted Image File
    print("\n[Test 6] Testing corrupted image file ...")
    corrupt_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorruptedcontent"
    resp_corrupt = client.post(
        "/predict/full",
        files={"file": ("corrupt.png", corrupt_bytes, "image/png")}
    )
    print(f"Status: {resp_corrupt.status_code} | Response: {resp_corrupt.json()}")
    assert resp_corrupt.status_code == 400, "Expected HTTP 400 for corrupt image!"
    results["corrupted_image"] = {
        "status_code": resp_corrupt.status_code,
        "response": resp_corrupt.json()
    }

    # Test 7: Large Image Canvas (3000x3000px)
    print("\n[Test 7] Testing oversized image canvas (3000x3000px) ...")
    large_img = Image.new("RGB", (3000, 3000), color=(128, 128, 128))
    buf = io.BytesIO()
    large_img.save(buf, format="PNG")
    large_bytes = buf.getvalue()

    t0 = time.time()
    resp_large = client.post(
        "/predict/full",
        files={"file": ("large_scan.png", large_bytes, "image/png")}
    )
    latency_large = time.time() - t0
    print(f"Status: {resp_large.status_code} | Latency: {latency_large:.4f}s")
    assert resp_large.status_code == 200, "Oversized image failed!"
    print(f"Original Resolution Detected: {resp_large.json()['original_resolution']}")
    results["large_image"] = {
        "status_code": resp_large.status_code,
        "latency_sec": round(latency_large, 4),
        "original_resolution": resp_large.json()["original_resolution"],
        "predicted_class": resp_large.json()["classification"]["predicted_class"]
    }

    # Test 8: Web GUI Static File
    print("\n[Test 8] Testing GET / (Clinical Web Workstation GUI) ...")
    resp_gui = client.get("/")
    print(f"Status: {resp_gui.status_code} | Content-Type: {resp_gui.headers.get('content-type')}")
    assert resp_gui.status_code == 200, "GUI index.html failed!"
    results["gui_endpoint"] = {
        "status_code": resp_gui.status_code,
        "content_type": resp_gui.headers.get("content-type"),
        "html_size_bytes": len(resp_gui.text)
    }

    # Save verification report
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/api_test_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAPI verification completed successfully! Report saved to {report_path}")

if __name__ == "__main__":
    run_api_verification()
