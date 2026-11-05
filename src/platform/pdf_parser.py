"""
PDF Container Parser
Extracts and renders pages from medical scan PDF containers (e.g. exported radiology reports/scans).
PDF functions strictly as an input container; text/OCR is never used as medical findings.
"""

import io
import base64
from typing import Dict, Any, List, Optional
from PIL import Image
import pypdfium2 as pdfium


class PDFParser:
    """
    Parses PDF containers, extracting rendered pages for operator inspection and modality analysis.
    """

    @classmethod
    def parse_pdf(cls, file_bytes: bytes, filename: str = "document.pdf", max_pages: int = 25) -> Dict[str, Any]:
        """
        Parses a PDF file, extracting page count and high-fidelity page previews as base64 images.
        """
        if not file_bytes or len(file_bytes) < 4:
            return {
                "status": "error",
                "error": "Unable to open this PDF.",
                "num_pages": 0,
                "pages": []
            }

        if not file_bytes.startswith(b"%PDF-"):
            return {
                "status": "error",
                "error": "Unable to open this PDF.",
                "num_pages": 0,
                "pages": []
            }

        try:
            doc = pdfium.PdfDocument(file_bytes)
        except Exception:
            return {
                "status": "error",
                "error": "Unable to open this PDF.",
                "num_pages": 0,
                "pages": []
            }

        num_pages = len(doc)
        if num_pages == 0:
            return {
                "status": "error",
                "error": "This PDF does not contain a supported medical image.",
                "num_pages": 0,
                "pages": []
            }

        pages: List[Dict[str, Any]] = []
        limit = min(num_pages, max_pages)

        for page_idx in range(limit):
            try:
                page = doc.get_page(page_idx)
                # Render at 1.5x scale for clear medical scan preview
                rendered_img = page.render(scale=1.5).to_pil()
                
                # Encode to PNG base64
                buf = io.BytesIO()
                rendered_img.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

                pages.append({
                    "page_number": page_idx + 1,
                    "width": rendered_img.width,
                    "height": rendered_img.height,
                    "preview_b64": b64_str
                })
            except Exception:
                continue

        if not pages:
            return {
                "status": "error",
                "error": "No supported medical image found in this PDF.",
                "num_pages": num_pages,
                "pages": []
            }

        return {
            "status": "success",
            "filename": filename,
            "num_pages": num_pages,
            "pages": pages
        }

    @classmethod
    def render_page_to_png_bytes(cls, file_bytes: bytes, page_number: int = 1) -> bytes:
        """
        Renders a specific 1-indexed page of a PDF directly to PNG byte stream.
        """
        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError("Unable to open this PDF.")

        try:
            doc = pdfium.PdfDocument(file_bytes)
        except Exception:
            raise ValueError("Unable to open this PDF.")

        if len(doc) == 0:
            raise ValueError("This PDF does not contain a supported medical image.")

        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"Page number {page_number} is out of range (1 to {len(doc)}).")

        page = doc.get_page(page_number - 1)
        pil_img = page.render(scale=2.0).to_pil().convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
