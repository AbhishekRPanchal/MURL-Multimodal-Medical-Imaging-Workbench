"""
NIfTI Volumetric Parser and Multi-Planar Slicer
Parses 3D and 4D NIfTI volumes (.nii, .nii.gz) using nibabel.
Extracts voxel spacing, orientation, intensity distributions,
and generates axial, coronal, and sagittal preview slices for frontend display.
"""

import io
import os
import base64
import tempfile
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import nibabel as nib


class NIfTIMetadataParser:
    """
    Parses volumetric 3D and 4D NIfTI medical scans.
    """

    @classmethod
    def parse_from_bytes(cls, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Safely writes NIfTI bytes to a temporary file, inspects header via nibabel,
        and extracts clinical orientation, voxel geometry, and orthogonal slices.
        """
        # Determine appropriate temp suffix (.nii.gz vs .nii)
        suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            return cls.parse_from_path(tmp_path, filename=filename)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @classmethod
    def parse_from_path(cls, file_path: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads NIfTI scan from disk and extracts volume metadata.
        """
        img = nib.load(file_path)
        header = img.header
        shape = list(img.shape)
        ndim = len(shape)

        # Voxel spacing
        zooms = [round(float(z), 3) for z in header.get_zooms()]

        # Anatomical orientation
        try:
            orientation = list(nib.aff2axcodes(img.affine))
        except Exception:
            orientation = ["UNKNOWN"]

        # Data type and units
        dtype_str = str(header.get_data_dtype())
        xyzt_units = header.get_xyzt_units()

        # Slice shape inspection
        data = img.get_fdata(dtype=np.float32)

        # Handle 4D volumes (e.g. multi-parametric MRI or functional series)
        if ndim == 4:
            vol_3d = data[..., 0]  # Take first sequence/timepoint for structural preview
            num_timepoints_or_channels = shape[3]
        else:
            vol_3d = data
            num_timepoints_or_channels = 1

        # Intensity statistics
        val_min = float(np.nanmin(vol_3d))
        val_max = float(np.nanmax(vol_3d))
        val_mean = float(np.nanmean(vol_3d))
        val_std = float(np.nanstd(vol_3d))

        # Multi-planar slice previews (orthogonal views)
        h, w, d = vol_3d.shape[:3]
        axial_mid = vol_3d[:, :, d // 2]
        coronal_mid = vol_3d[:, w // 2, :]
        sagittal_mid = vol_3d[h // 2, :, :]

        def normalize_slice(s: np.ndarray) -> Image.Image:
            # Percentile windowing to handle high dynamic range / CT outliers
            p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
            clipped = np.clip(s, p1, p99)
            norm = (clipped - p1) / (p99 - p1 + 1e-8) * 255.0
            # Transpose if needed to match radiographic viewing conventions
            return Image.fromarray(norm.astype(np.uint8).T)

        def render_slice_b64(s: np.ndarray) -> str:
            pil = normalize_slice(s)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # Multi-planar slice stacks for interactive MPR browsing (capped at 64 samples per axis)
        max_stack = 64
        step_z = max(1, d // max_stack) if d > max_stack else 1
        step_y = max(1, w // max_stack) if w > max_stack else 1
        step_x = max(1, h // max_stack) if h > max_stack else 1

        slice_stacks = {
            "axial": [render_slice_b64(vol_3d[:, :, z]) for z in range(0, d, step_z)],
            "coronal": [render_slice_b64(vol_3d[:, y, :]) for y in range(0, w, step_y)],
            "sagittal": [render_slice_b64(vol_3d[x, :, :]) for x in range(0, h, step_x)],
            "total_counts": {"axial": d, "coronal": w, "sagittal": h}
        }

        return {
            "format": "nifti",
            "filename": filename or os.path.basename(file_path),
            "dimensions": shape,
            "spatial_ndim": ndim,
            "num_channels_or_volumes": num_timepoints_or_channels,
            "voxel_spacing_mm": zooms,
            "anatomical_orientation": orientation,
            "affine": img.affine.tolist(),
            "data_type": dtype_str,
            "units": xyzt_units,
            "intensity_distribution": {
                "min": round(val_min, 2),
                "max": round(val_max, 2),
                "mean": round(val_mean, 2),
                "std": round(val_std, 2)
            },
            "preview_slices": {
                "axial": normalize_slice(axial_mid),
                "coronal": normalize_slice(coronal_mid),
                "sagittal": normalize_slice(sagittal_mid)
            },
            "slice_stacks": slice_stacks
        }
