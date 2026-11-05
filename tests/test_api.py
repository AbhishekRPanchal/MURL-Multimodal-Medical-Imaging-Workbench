"""
Unit Tests for FastAPI REST Endpoints
"""

import io
from PIL import Image
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def create_dummy_image_bytes():
    arr = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "device" in data

def test_predict_full_endpoint():
    img_buf = create_dummy_image_bytes()
    resp = client.post(
        "/predict/full",
        files={"file": ("test.png", img_buf, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "original_resolution" in data
    assert "classification" in data
    assert "segmentation" in data

def test_invalid_file_type():
    resp = client.post(
        "/predict/full",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )
    assert resp.status_code == 400
