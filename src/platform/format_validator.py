"""
Image Format Validator
Validates file extensions, magic bytes headers, and file size constraints
for medical imaging multi-modal pipelines (PNG, JPEG, DICOM, NIfTI).
"""

import os
from typing import Dict, Any, Tuple, Optional

# Standard medical and raster formats
SUPPORTED_EXTENSIONS = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".dcm": "dicom",
    ".dicom": "dicom",
    ".nii": "nifti",
    ".nii.gz": "nifti_gz",
    ".pdf": "pdf"
}

# Maximum allowed file sizes in bytes
MAX_2D_SIZE_BYTES = 50 * 1024 * 1024    # 50 MB
MAX_3D_SIZE_BYTES = 250 * 1024 * 1024   # 250 MB


class ImageFormatValidator:
    """
    Validates medical image files prior to model ingestion.
    Checks file integrity, magic byte headers, and enforces size constraints.
    """

    @staticmethod
    def inspect_magic_bytes(header: bytes) -> Optional[str]:
        """
        Inspect the binary header signature to detect actual file format.
        """
        if len(header) < 4:
            return None

        # PNG signature: 89 50 4E 47 0D 0A 1A 0A
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"

        # JPEG signature: FF D8 FF
        if header.startswith(b"\xff\xd8\xff"):
            return "jpeg"

        # PDF signature: %PDF- (25 50 44 46 2D)
        if header.startswith(b"%PDF-"):
            return "pdf"

        # Gzip signature (e.g. .nii.gz): 1F 8B
        if header.startswith(b"\x1f\x8b"):
            return "nifti_gz"

        # DICOM signature: 'DICM' starting at byte offset 128
        if len(header) >= 132 and header[128:132] == b"DICM":
            return "dicom"

        # Uncompressed NIfTI-1: sizeof_hdr == 348 (0x0000015C in little-endian: 5C 01 00 00, or big-endian: 00 00 01 5C)
        if len(header) >= 4:
            sizeof_hdr_le = int.from_bytes(header[0:4], byteorder="little")
            sizeof_hdr_be = int.from_bytes(header[0:4], byteorder="big")
            if sizeof_hdr_le == 348 or sizeof_hdr_be == 348:
                return "nifti"

        return None

    @classmethod
    def validate(cls, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Comprehensive validation of medical image binary stream and filename.

        Args:
            file_bytes: Raw binary content of the file.
            filename: Original file name or path.

        Returns:
            Dictionary containing validation status, detected format, file size, and error details if any.
        """
        if not file_bytes or len(file_bytes) == 0:
            return {
                "is_valid": False,
                "error": "File is empty (0 bytes).",
                "format": None,
                "file_size_bytes": 0
            }

        file_size = len(file_bytes)
        lower_name = filename.lower()

        # Determine extension
        detected_ext = None
        for ext in [".nii.gz", ".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".nii", ".pdf"]:
            if lower_name.endswith(ext):
                detected_ext = ext
                break

        if not detected_ext:
            return {
                "is_valid": False,
                "error": f"Unsupported file extension in '{filename}'. Supported extensions: {list(SUPPORTED_EXTENSIONS.keys())}",
                "format": None,
                "file_size_bytes": file_size
            }

        expected_format = SUPPORTED_EXTENSIONS[detected_ext]

        # Enforce size constraints
        is_3d = expected_format in ["nifti", "nifti_gz"]
        max_size = MAX_3D_SIZE_BYTES if is_3d else MAX_2D_SIZE_BYTES
        if file_size > max_size:
            return {
                "is_valid": False,
                "error": f"File size exceeds safety limit ({file_size} bytes > {max_size} bytes limit).",
                "format": expected_format,
                "file_size_bytes": file_size
            }

        # Validate magic header
        magic_format = cls.inspect_magic_bytes(file_bytes[:352])

        # Allow DICOM files without standard preamble (some raw DICOMs omit the 128-byte preamble)
        if expected_format == "dicom" and magic_format is None:
            # Check for DICOM group tag 0x0002 or 0x0008 in first bytes
            if len(file_bytes) >= 4 and (file_bytes[:2] in [b"\x02\x00", b"\x08\x00", b"\x00\x02", b"\x00\x08"]):
                magic_format = "dicom"

        if magic_format is not None and magic_format != expected_format:
            # Special case: jpeg format can match .jpg or .jpeg
            if not (expected_format == "jpeg" and magic_format == "jpeg"):
                return {
                    "is_valid": False,
                    "error": f"File header mismatch: extension suggests '{expected_format}', but header magic indicates '{magic_format}'.",
                    "format": magic_format,
                    "file_size_bytes": file_size
                }

        # Validate that formats with mandatory magic headers actually matched
        if magic_format is None and expected_format in ["png", "jpeg", "nifti_gz", "nifti", "pdf"]:
            return {
                "is_valid": False,
                "error": f"Invalid or corrupted {expected_format.upper()} scan: file header signature does not match format specification." if expected_format != "pdf" else "Unable to open this PDF.",
                "format": expected_format,
                "file_size_bytes": file_size
            }

        resolved_format = magic_format or expected_format

        return {
            "is_valid": True,
            "error": None,
            "format": resolved_format,
            "extension": detected_ext,
            "file_size_bytes": file_size,
            "is_volumetric": is_3d
        }
