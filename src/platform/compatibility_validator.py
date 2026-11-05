"""
Model Compatibility Validator
Verifies that uploaded scan attributes (modality, dimensionality, channels,
MRI sequences, file format) satisfy the strict requirements of target models.
"""

from typing import Dict, Any, List, Optional


class ModelCompatibilityValidator:
    """
    Ensures input data conforms to model specifications before inference is initiated.
    Prevents silent failures, shape mismatches, and inappropriate cross-modality inference.
    """

    @staticmethod
    def validate_compatibility(
        model_metadata: Dict[str, Any],
        scan_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates scan properties against a model's registered requirements.

        Args:
            model_metadata: Model entry dictionary from ModelRegistry.
            scan_properties: Extracted scan metadata, including:
                - modality: str
                - format: str (e.g., 'png', 'dicom', 'nifti')
                - spatial_ndim: int (2 for 2D, 3 for 3D)
                - num_channels: int
                - mri_sequences: Optional[List[str]]

        Returns:
            Dict containing 'compatible': bool, 'issues': List[str], 'warnings': List[str]
        """
        issues: List[str] = []
        warnings: List[str] = []

        model_id = model_metadata.get("model_id", "unknown")
        req_modality = model_metadata.get("modality", "")
        req_formats = [fmt.lower() for fmt in model_metadata.get("supported_image_formats", [])]
        req_dims = model_metadata.get("required_input_dimensions", [])
        req_channels = model_metadata.get("num_input_channels", 1)
        req_sequences = [s.upper() for s in model_metadata.get("required_mri_sequences", [])]

        scan_modality = scan_properties.get("modality", "")
        scan_format = str(scan_properties.get("format", "")).lower()
        scan_ndim = scan_properties.get("spatial_ndim", 2)
        scan_channels = scan_properties.get("num_channels", 1)
        scan_sequences = [s.upper() for s in scan_properties.get("mri_sequences", [])]

        # 1. Modality check
        if scan_modality == "Segmentation Mask":
            issues.append(
                f"Input file is a segmentation label mask, not an original imaging acquisition volume. "
                f"Model '{model_id}' requires original {req_modality} data."
            )
            return {"compatible": False, "issues": issues, "warnings": warnings, "model_id": model_id}

        modality_compatible = False
        if not scan_modality:
            modality_compatible = True
        elif req_modality.lower() == scan_modality.lower():
            modality_compatible = True
        elif req_modality.lower() == "multi-modal":
            modality_compatible = True
        elif ("mri/ct" in scan_modality.lower() or "unknown / volumetric" in scan_modality.lower()) and req_modality.lower() in ["ct", "mri"]:
            modality_compatible = True
            warnings.append(f"Volumetric scan routed to '{model_id}' under {req_modality} protocol.")
        elif req_modality.lower() in scan_modality.lower():
            modality_compatible = True

        if not modality_compatible:
            issues.append(
                f"Modality mismatch: Model '{model_id}' requires {req_modality}, but input scan was detected as {scan_modality}."
            )

        # 1b. Anatomical region check (prevent mixing e.g. Breast and Thyroid)
        req_anatomy = model_metadata.get("anatomical_region", "")
        scan_anatomy = scan_properties.get("anatomical_region", "")
        if req_anatomy and scan_anatomy and req_anatomy.lower() != scan_anatomy.lower() and req_anatomy.lower() != "whole-body":
            issues.append(
                f"Anatomical region mismatch: Model '{model_id}' is calibrated for '{req_anatomy}', but scan is '{scan_anatomy}'."
            )

        # 2. Format check
        format_aliases = [scan_format]
        if scan_format in ["jpeg", "jpg"]:
            format_aliases.extend(["jpg", "jpeg"])
        if scan_format in ["nifti", "nii"]:
            format_aliases.extend(["nii", "nifti"])
        if scan_format in ["nifti_gz", "nii.gz"]:
            format_aliases.extend(["nii.gz", "nii", "nifti", "nifti_gz"])
        if scan_format in ["dicom", "dcm"]:
            format_aliases.extend(["dcm", "dicom"])

        if scan_format and not any(alias in req_formats for alias in format_aliases):
            issues.append(
                f"Format mismatch: Model '{model_id}' supports {req_formats}, but scan format is '{scan_format}'."
            )

        # 3. Spatial dimensionality check (2D vs 3D)
        model_spatial_ndim = len(req_dims)
        if scan_ndim != model_spatial_ndim:
            issues.append(
                f"Dimensionality mismatch: Model '{model_id}' requires {model_spatial_ndim}D input (dims: {req_dims}), "
                f"but scan provides {scan_ndim}D data."
            )

        # 4. MRI sequences check (if required)
        if req_sequences:
            missing_seqs = [s for s in req_sequences if s not in scan_sequences]
            if missing_seqs:
                if scan_ndim == 3 or scan_channels == 1:
                    warnings.append(
                        f"Sequence advisory: Multi-parametric MRI model recommends {req_sequences}. "
                        f"Single sequence scan will be previewed in multi-planar reconstruction."
                    )
                else:
                    issues.append(
                        f"Missing required MRI sequences for '{model_id}': Required {req_sequences}, "
                        f"but scan is missing: {missing_seqs}."
                    )

        # 5. Channel count check
        if scan_channels != req_channels:
            if req_channels == 3 and scan_channels == 1:
                warnings.append(
                    f"Channel expansion: Model '{model_id}' expects 3 channels. 1-channel input will be replicated to RGB."
                )
            elif req_channels == 1 and scan_channels == 3:
                warnings.append(
                    f"Channel reduction: Model '{model_id}' expects 1 channel. RGB input will be converted to grayscale."
                )
            elif req_channels == 4 and scan_channels == 1:
                warnings.append(
                    f"Channel replication: Model '{model_id}' expects 4 sequence channels. "
                    f"1-channel volume will be replicated across structural channels for preview."
                )
            else:
                issues.append(
                    f"Channel mismatch: Model requires {req_channels} channels, but scan provides {scan_channels}."
                )

        return {
            "compatible": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "model_id": model_id
        }
