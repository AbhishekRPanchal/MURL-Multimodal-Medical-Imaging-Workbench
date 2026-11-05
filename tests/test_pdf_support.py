"""
PDF Support Test Suite
Verifies PDF container handling, multi-page extraction, page rendering,
modality detection on selected pages, analysis execution, and error handling.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app
from src.platform.format_validator import ImageFormatValidator
from src.platform.pdf_parser import PDFParser

client = TestClient(app)


def create_test_pdf(num_pages: int = 2) -> bytes:
    """Helper creating an in-memory multi-page medical PDF."""
    imgs = [
        Image.new("RGB", (200 + i * 50, 200 + i * 50), color=(i * 60, 100, 150))
        for i in range(num_pages)
    ]
    buf = io.BytesIO()
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


class TestPDFSupport:

    def test_pdf_format_validation(self):
        pdf_bytes = create_test_pdf(1)
        res = ImageFormatValidator.validate(pdf_bytes, "radiology_scan.pdf")
        assert res["is_valid"] is True
        assert res["format"] == "pdf"

    def test_pdf_parser_page_extraction(self):
        pdf_bytes = create_test_pdf(3)
        res = PDFParser.parse_pdf(pdf_bytes, "report.pdf")
        assert res["status"] == "success"
        assert res["num_pages"] == 3
        assert len(res["pages"]) == 3
        assert res["pages"][0]["page_number"] == 1
        assert "preview_b64" in res["pages"][0]

    def test_pdf_parser_page_render(self):
        pdf_bytes = create_test_pdf(2)
        png_bytes = PDFParser.render_page_to_png_bytes(pdf_bytes, page_number=2)
        assert png_bytes.startswith(b"\x89PNG")
        img = Image.open(io.BytesIO(png_bytes))
        assert img.width > 0
        assert img.height > 0

    def test_pdf_inspect_api(self):
        pdf_bytes = create_test_pdf(2)
        response = client.post(
            "/api/inspect",
            files={"file": ("scan_container.pdf", pdf_bytes, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["format"] == "pdf"
        assert data["pdf_info"] is not None
        assert data["pdf_info"]["num_pages"] == 2
        assert len(data["pdf_info"]["pages"]) == 2

    def test_pdf_page_selection_api(self):
        pdf_bytes = create_test_pdf(3)
        response = client.post(
            "/api/pdf/page",
            data={"page_number": 2},
            files={"file": ("scan_container.pdf", pdf_bytes, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["page_number"] == 2
        assert "image_b64" in data
        assert "modality_detection" in data

    def test_pdf_analysis_execution_api(self):
        # Create a PDF with a CXR scan as page 1
        cxr_img = Image.open("MURL_DEMO_DATA/chest_xray/cxr_normal_02.png").convert("RGB")
        buf = io.BytesIO()
        cxr_img.save(buf, format="PDF")
        pdf_bytes = buf.getvalue()

        response = client.post(
            "/api/analyze",
            data={"page_number": 1},
            files={"file": ("cxr_patient_report.pdf", pdf_bytes, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["active_modality"] == "X-ray"
        assert "prediction" in data

    def test_invalid_pdf_error_handling(self):
        corrupt_bytes = b"not a real pdf header or content"
        res = PDFParser.parse_pdf(corrupt_bytes, "corrupt.pdf")
        assert res["status"] == "error"
        assert "Unable to open this PDF." in res["error"]

        response = client.post(
            "/api/inspect",
            files={"file": ("corrupt.pdf", corrupt_bytes, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Unable to open this PDF." in data["error"]

    def test_out_of_range_page_handling(self):
        pdf_bytes = create_test_pdf(1)
        response = client.post(
            "/api/pdf/page",
            data={"page_number": 99},
            files={"file": ("scan.pdf", pdf_bytes, "application/pdf")}
        )
        assert response.status_code == 400
        assert "out of range" in response.json()["detail"].lower()
