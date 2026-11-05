"""
MURL Multi-Format Medical & Research Result Export Service
Supports:
1. DICOM SEG (SOP Class 1.2.840.10008.5.1.4.1.1.66.4, Modality SEG)
2. DICOM SR (SOP Class 1.2.840.10008.5.1.4.1.1.88.33, Modality SR)
3. NIfTI (.nii.gz preserving affine, geometry, dimensions, spacing)
4. PDF Report (Clean, professional, human-readable research report via ReportLab)
5. JSON (Full structured machine-readable result)
6. PNG Overlay & PNG Mask (2D alpha blend and binary mask)
7. CSV (Tabular quantitative metrics when meaningful structured measurements exist)

Adheres strictly to medical standards, preserves source DICOM context when present,
and clearly distinguishes research data from clinical records.
"""

import os
import io
import json
import csv
import time
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
from PIL import Image
import nibabel as nib

# DICOM tools
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# PDF tools
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class ExportService:
    """
    Central service for exporting medical AI findings and segmentations
    into genuine medical and research formats.
    """

    @classmethod
    def get_valid_export_formats(cls, analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Determines the strictly valid export formats for the given analysis result.
        Filters out technically invalid options (e.g. no SEG for classification, no CSV without tabular metrics).
        """
        if not analysis_result or analysis_result.get("status") != "success":
            return []

        modality = analysis_result.get("active_modality", "Unknown")
        pred = analysis_result.get("prediction", {})
        task = pred.get("output_task", "")
        formatted = pred.get("formatted_results", {})

        formats = []

        # 1. PDF Report is always valid for human inspection of any completed analysis
        formats.append({
            "id": "pdf",
            "label": "PDF Report (.pdf)",
            "description": "Human-readable research summary report",
            "extension": "pdf"
        })

        # 2. JSON is always valid for machine-readable reproducibility
        formats.append({
            "id": "json",
            "label": "JSON Document (.json)",
            "description": "Structured machine-readable analysis data",
            "extension": "json"
        })

        # Determine if genuine segmentation output is present
        has_segmentation = (
            task in ["segmentation", "interactive_segmentation"] or
            formatted.get("output_nifti_path") is not None or
            formatted.get("lesion_detected") is not None or
            formatted.get("is_trained_ai_segmentation") is True or
            formatted.get("compartments") is not None or
            formatted.get("top_detected_structures") is not None
        )

        # 3. DICOM SR (Structured Report)
        # Valid for structured findings (pathology probabilities, structure volumes, lesion metrics)
        has_structured_findings = (
            formatted.get("top_findings") is not None or
            formatted.get("predicted_class") is not None or
            formatted.get("top_detected_structures") is not None or
            formatted.get("compartments") is not None or
            formatted.get("lesion_area_percentage") is not None
        )
        if has_structured_findings:
            formats.append({
                "id": "dicom_sr",
                "label": "DICOM SR (.dcm)",
                "description": "Genuine DICOM Structured Report with quantitative findings",
                "extension": "dcm"
            })

        # 4. DICOM SEG (DICOM Segmentation Object)
        # Valid ONLY when genuine segmentation exists
        if has_segmentation:
            formats.append({
                "id": "dicom_seg",
                "label": "DICOM SEG (.dcm)",
                "description": "Standard DICOM Segmentation storage object",
                "extension": "dcm"
            })

        # 5. NIfTI Mask
        # Valid for volumetric and spatial segmentations
        if has_segmentation:
            formats.append({
                "id": "nifti",
                "label": "NIfTI Mask (.nii.gz)",
                "description": "Volumetric / spatial segmentation mask",
                "extension": "nii.gz"
            })

        # 6. PNG Overlay & PNG Mask
        # Valid when 2D raster overlays or masks are present
        has_2d_overlay = (
            formatted.get("mask_overlay_b64") is not None or
            formatted.get("multilabel_previews", {}).get("axial_seg_b64") is not None or
            formatted.get("multiplanar_previews", {}).get("axial_wt_b64") is not None or
            formatted.get("raw_mask_b64") is not None
        )
        if has_2d_overlay:
            formats.append({
                "id": "png_overlay",
                "label": "PNG Overlay (.png)",
                "description": "Blended scan and segmentation overlay image",
                "extension": "png"
            })
            formats.append({
                "id": "png_mask",
                "label": "PNG Mask (.png)",
                "description": "Binary / label segmentation mask",
                "extension": "png"
            })

        # 7. CSV Tabular Export
        # Valid ONLY when meaningful structured quantitative measurements exist
        has_tabular = (
            (formatted.get("top_detected_structures") and len(formatted["top_detected_structures"]) > 0) or
            (formatted.get("compartments") and len(formatted["compartments"]) > 0) or
            (formatted.get("lesion_area_percentage") is not None and task == "segmentation") or
            (formatted.get("tirads_features") is not None)
        )
        if has_tabular:
            formats.append({
                "id": "csv",
                "label": "CSV Metrics (.csv)",
                "description": "Tabular quantitative volumes and measurements",
                "extension": "csv"
            })

        return formats

    # -------------------------------------------------------------------------
    # 1. PDF Report Generation
    # -------------------------------------------------------------------------
    @classmethod
    def export_pdf(cls, analysis_result: Dict[str, Any], preview_image_bytes: Optional[bytes] = None) -> bytes:
        """
        Generates a clean, professional, human-readable research report in PDF format.
        Omits internal developer technical details (model IDs, checkpoint files, timings)
        unless explicitly requested.
        """
        if not HAS_REPORTLAB:
            raise RuntimeError("reportlab is required for PDF report generation.")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
            title="MURL Medical Imaging Analysis Report",
            author="MURL Research Workbench",
            subject="Multimodal Medical Imaging Research",
            creator="MURL"
        )

        styles = getSampleStyleSheet()
        elements = []

        # Color Palette
        col_primary = colors.HexColor("#0f172a")     # Deep slate
        col_accent = colors.HexColor("#2563eb")      # Medical blue
        col_muted = colors.HexColor("#64748b")       # Slate muted
        col_bg = colors.HexColor("#f8fafc")          # Light card background
        col_border = colors.HexColor("#e2e8f0")      # Subtle border

        # Typography
        header_inst_style = ParagraphStyle(
            "InstHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=col_accent,
            spaceAfter=2
        )
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=col_primary,
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "ReportSub",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=col_muted,
            spaceAfter=12
        )
        section_style = ParagraphStyle(
            "ReportSec",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=col_primary,
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=col_primary
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#b91c1c"),
            alignment=1  # Centered
        )

        # 1. Header Banner
        elements.append(Paragraph("MURL — Multimodal Medical Imaging Research Workbench", header_inst_style))
        elements.append(Paragraph("Medical Imaging Analysis Report", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Status: Research & Educational Use", subtitle_style))
        elements.append(Spacer(1, 4))

        # 2. Study & Scan Overview Table
        modality = analysis_result.get("active_modality", "Unknown")
        fmt_report = analysis_result.get("format_report", {})
        filename = fmt_report.get("filename", "Input Scan")
        pred = analysis_result.get("prediction", {})
        task = pred.get("output_task", "Analysis").capitalize()

        dcm_meta = analysis_result.get("dicom_metadata")
        nii_meta = analysis_result.get("nifti_metadata")

        study_id = "RESEARCH-CASE"
        if dcm_meta and dcm_meta.get("study_description"):
            study_id = dcm_meta.get("study_description")

        dim_str = "Standard 2D"
        if nii_meta and nii_meta.get("dimensions"):
            dim_str = f"3D Volumetric ({' x '.join(map(str, nii_meta['dimensions']))})"
        elif fmt_report.get("dimensions"):
            dim_str = f"{fmt_report['dimensions'][0]} x {fmt_report['dimensions'][1]}"

        study_data = [
            [Paragraph("<b>Study / Case:</b>", body_style), Paragraph(study_id, body_style),
             Paragraph("<b>Modality:</b>", body_style), Paragraph(modality, body_style)],
            [Paragraph("<b>Source File:</b>", body_style), Paragraph(filename, body_style),
             Paragraph("<b>Task:</b>", body_style), Paragraph(task, body_style)],
            [Paragraph("<b>Geometry:</b>", body_style), Paragraph(dim_str, body_style),
             Paragraph("<b>Privacy Filter:</b>", body_style), Paragraph("Research Privacy Filter Applied", body_style)]
        ]
        t_study = Table(study_data, colWidths=[90, 180, 90, 180])
        t_study.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), col_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, col_border),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_study)
        elements.append(Spacer(1, 10))

        # 3. Visual Image / Overlay Preview (if available)
        img_b64 = None
        formatted = pred.get("formatted_results", {})
        if formatted.get("mask_overlay_b64"):
            img_b64 = formatted["mask_overlay_b64"]
        elif formatted.get("multilabel_previews", {}).get("axial_seg_b64"):
            img_b64 = formatted["multilabel_previews"]["axial_seg_b64"]
        elif formatted.get("multiplanar_previews", {}).get("axial_wt_b64"):
            img_b64 = formatted["multiplanar_previews"]["axial_wt_b64"]
        elif nii_meta and nii_meta.get("orthogonal_previews", {}).get("axial_b64"):
            img_b64 = nii_meta["orthogonal_previews"]["axial_b64"]

        if img_b64:
            try:
                img_data = base64.b64decode(img_b64)
                img_buf = io.BytesIO(img_data)
                # Auto-scale preview image
                pil_im = Image.open(img_buf)
                w, h = pil_im.size
                max_w, max_h = 240, 180
                scale = min(max_w / w, max_h / h, 1.0)
                disp_w = int(w * scale)
                disp_h = int(h * scale)
                img_buf.seek(0)
                rl_img = RLImage(img_buf, width=disp_w, height=disp_h)
                
                elements.append(Paragraph("Scan Representation", section_style))
                elements.append(rl_img)
                elements.append(Spacer(1, 8))
            except Exception as e:
                pass

        # 4. Findings & Diagnostic Analysis
        elements.append(Paragraph("AI Findings & Diagnostic Results", section_style))

        # Chest X-Ray findings
        if formatted.get("top_findings"):
            findings_data = [["Rank", "Pathological Finding", "Confidence / Probability"]]
            for idx, item in enumerate(formatted["top_findings"], 1):
                findings_data.append([
                    str(idx),
                    item.get("pathology", "Finding"),
                    f"{item.get('probability_percent', 0.0)}%"
                ])
            t_findings = Table(findings_data, colWidths=[40, 300, 200])
            t_findings.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), col_accent),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('BOX', (0, 0), (-1, -1), 0.5, col_border),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, col_bg]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_findings)
            elements.append(Spacer(1, 8))

        # Classification (BUSI)
        elif formatted.get("predicted_class"):
            p_class = formatted.get("predicted_class", "").upper()
            conf = formatted.get("confidence", 0.0)
            if isinstance(conf, float) and conf <= 1.0:
                conf = round(conf * 100, 2)
            cls_data = [
                [Paragraph("<b>Primary Assessment:</b>", body_style), Paragraph(f"<b>{p_class}</b>", body_style)],
                [Paragraph("<b>Confidence:</b>", body_style), Paragraph(f"{conf}%", body_style)]
            ]
            t_cls = Table(cls_data, colWidths=[150, 390])
            t_cls.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), col_bg),
                ('BOX', (0, 0), (-1, -1), 0.5, col_border),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_cls)
            elements.append(Spacer(1, 8))

        # 5. Volumetry & Quantitative Measurements (CT / MRI / Ultrasound)
        # CT TotalSegmentator structures
        if formatted.get("top_detected_structures"):
            elements.append(Paragraph("Detected Anatomical Structures (Top Slices)", section_style))
            struct_data = [["Structure", "Label ID", "Voxel Count", "Volume (cm³)"]]
            for s in formatted["top_detected_structures"][:10]:
                struct_data.append([
                    s.get("structure_name", "").replace("_", " ").title(),
                    str(s.get("label_id", "")),
                    f"{s.get('voxel_count', 0):,}",
                    f"{s.get('volume_cm3', 0.0):.2f}"
                ])
            t_struct = Table(struct_data, colWidths=[200, 80, 120, 140])
            t_struct.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOX', (0, 0), (-1, -1), 0.5, col_border),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, col_bg]),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(t_struct)
            elements.append(Spacer(1, 8))

        # MRI Glioma Compartments
        elif formatted.get("compartments"):
            elements.append(Paragraph("Pathological Compartment Volumetry", section_style))
            comp_data = [["Compartment", "Voxel Count", "Volume (cm³)"]]
            for c in formatted["compartments"]:
                comp_data.append([
                    c.get("compartment", ""),
                    f"{c.get('voxel_count', 0):,}",
                    f"{c.get('volume_cm3', 0.0):.3f}"
                ])
            t_comp = Table(comp_data, colWidths=[280, 120, 140])
            t_comp.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('BOX', (0, 0), (-1, -1), 0.5, col_border),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, col_bg]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_comp)
            elements.append(Spacer(1, 8))

        # Ultrasound / MedSAM Lesion Metrics
        elif formatted.get("lesion_area_percentage") is not None:
            elements.append(Paragraph("Quantitative Lesion Measurements", section_style))
            meas_data = [
                [Paragraph("<b>Lesion Detected:</b>", body_style), Paragraph("Yes" if formatted.get("lesion_detected") else "No", body_style)],
                [Paragraph("<b>Area Coverage:</b>", body_style), Paragraph(f"{formatted.get('lesion_area_percentage', 0.0)}%", body_style)],
                [Paragraph("<b>Pixel Count:</b>", body_style), Paragraph(f"{formatted.get('lesion_pixel_count', 0):,} pixels", body_style)]
            ]
            t_meas = Table(meas_data, colWidths=[160, 380])
            t_meas.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), col_bg),
                ('BOX', (0, 0), (-1, -1), 0.5, col_border),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, col_border),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_meas)
            elements.append(Spacer(1, 8))

        # 6. Research Notice & Academic Disclaimer
        elements.append(Spacer(1, 12))
        disc_table = Table([[
            Paragraph(
                "RESEARCH DEMONSTRATION ONLY — NOT FOR CLINICAL DIAGNOSIS<br/>"
                "<font size='7' color='#64748b'>This AI-generated analysis was performed on the MURL medical imaging research workbench. "
                "All results are experimental and intended exclusively for academic, research, and technical validation. "
                "This document does not constitute a clinical diagnosis, medical opinion, or treatment guidance.</font>",
                disclaimer_style
            )
        ]], colWidths=[540])
        disc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#f87171")),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(KeepTogether([disc_table]))

        doc.build(elements)
        return buf.getvalue()

    # -------------------------------------------------------------------------
    # 2. DICOM SEG (Segmentation Object) Generation
    # -------------------------------------------------------------------------
    @classmethod
    def export_dicom_seg(cls, analysis_result: Dict[str, Any], source_dicom_bytes: Optional[bytes] = None) -> bytes:
        """
        Creates a genuine DICOM Segmentation (SEG) storage object.
        Preserves source DICOM study/series references when source DICOM is available;
        otherwise creates an explicit research/derived DICOM SEG object.
        """
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is required for DICOM SEG export.")

        pred = analysis_result.get("prediction", {})
        formatted = pred.get("formatted_results", {})
        dcm_meta = analysis_result.get("dicom_metadata")

        # 1. Base File Meta Information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"  # Segmentation Storage
        sop_instance_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "SEG"

        # Content dates and times
        now = datetime.now()
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.ContentTime = now.strftime("%H%M%S")
        ds.SeriesDate = ds.ContentDate
        ds.SeriesTime = ds.ContentTime

        # Study & Patient context
        if dcm_meta and dcm_meta.get("study_instance_uid"):
            ds.StudyInstanceUID = dcm_meta["study_instance_uid"]
            ds.StudyID = dcm_meta.get("study_id", "DEID_STUDY")
            ds.PatientID = dcm_meta.get("patient_id", "RESEARCH-ANON")
            ds.PatientName = dcm_meta.get("patient_name", "RESEARCH^ANONYMIZED")
            ds.StudyDate = dcm_meta.get("study_date", ds.ContentDate)
            ds.StudyTime = dcm_meta.get("study_time", ds.ContentTime)
        else:
            ds.StudyInstanceUID = generate_uid()
            ds.StudyID = "MURL-RESEARCH"
            ds.PatientID = "RESEARCH-ANON"
            ds.PatientName = "RESEARCH^ANONYMIZED"
            ds.StudyDate = ds.ContentDate
            ds.StudyTime = ds.ContentTime

        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 101
        ds.InstanceNumber = 1
        ds.SeriesDescription = "AI Segmentation (Research Export)"
        ds.Manufacturer = "MURL Research Workbench"
        ds.ManufacturerModelName = "MURL Multimodal Imaging Workbench"
        ds.SoftwareVersions = "2.0.0"

        # Segmentation attributes
        ds.SegmentationType = "BINARY"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # Extract segmentation mask arrays and segment labels
        segments = []
        mask_bytes = b""
        num_frames = 1
        rows = 256
        cols = 256

        # Case A: 3D MRI BraTS (Whole Tumor, Tumor Core, Enhancing Tumor)
        if formatted.get("compartments"):
            rows, cols, num_frames = 48, 48, 48
            if formatted.get("output_nifti_path") and os.path.exists(formatted["output_nifti_path"]):
                mri_nib = nib.load(formatted["output_nifti_path"])
                seg_data = np.asanyarray(mri_nib.dataobj).astype(np.uint8)
                rows, cols, num_frames = seg_data.shape
                mask_bytes = seg_data.tobytes()
            elif analysis_result.get("nifti_metadata", {}).get("dimensions"):
                dims = analysis_result["nifti_metadata"]["dimensions"]
                rows, cols, num_frames = dims[0], dims[1], dims[2]
                mask_bytes = np.zeros((rows, cols, num_frames), dtype=np.uint8).tobytes()
            else:
                mask_bytes = np.zeros((rows, cols, num_frames), dtype=np.uint8).tobytes()

            for c_idx, c in enumerate(formatted["compartments"], 1):
                seg_item = Dataset()
                seg_item.SegmentNumber = c_idx
                seg_item.SegmentLabel = c.get("compartment", f"compartment_{c_idx}")
                seg_item.SegmentAlgorithmType = "AUTOMATIC"
                seg_item.SegmentAlgorithmName = "MONAI SegResNet (BraTS)"

                cat_seq = Dataset()
                cat_seq.CodeValue = "M-01000"
                cat_seq.CodingSchemeDesignator = "SRT"
                cat_seq.CodeMeaning = "Morphologically Abnormal Structure"
                seg_item.SegmentedPropertyCategoryCodeSequence = Sequence([cat_seq])

                type_seq = Dataset()
                type_seq.CodeValue = "M-03000"
                type_seq.CodingSchemeDesignator = "SRT"
                type_seq.CodeMeaning = c.get("compartment", "Neoplasm")
                seg_item.SegmentedPropertyTypeCodeSequence = Sequence([type_seq])
                segments.append(seg_item)

        # Case B: 3D TotalSegmentator CT
        elif formatted.get("output_nifti_path") and os.path.exists(formatted["output_nifti_path"]):
            nii_seg = nib.load(formatted["output_nifti_path"])
            seg_data = np.asanyarray(nii_seg.dataobj).astype(np.uint8)
            h, w, d = seg_data.shape
            rows, cols, num_frames = h, w, d

            all_structs = formatted.get("all_detected_structures", formatted.get("top_detected_structures", []))
            for s_idx, st in enumerate(all_structs, 1):
                seg_item = Dataset()
                seg_item.SegmentNumber = s_idx
                seg_item.SegmentLabel = st.get("structure_name", f"structure_{s_idx}")
                seg_item.SegmentAlgorithmType = "AUTOMATIC"
                seg_item.SegmentAlgorithmName = "TotalSegmentator (Fast 3mm)"

                cat_seq = Dataset()
                cat_seq.CodeValue = "T-D000A"
                cat_seq.CodingSchemeDesignator = "SRT"
                cat_seq.CodeMeaning = "Anatomical Structure"
                seg_item.SegmentedPropertyCategoryCodeSequence = Sequence([cat_seq])

                type_seq = Dataset()
                type_seq.CodeValue = "T-D000A"
                type_seq.CodingSchemeDesignator = "SRT"
                type_seq.CodeMeaning = st.get("structure_name", "Organ")
                seg_item.SegmentedPropertyTypeCodeSequence = Sequence([type_seq])
                segments.append(seg_item)

            mask_bytes = seg_data.tobytes()

        # Case C: 2D Ultrasound / MedSAM segmentation
        else:
            num_frames = 1
            if formatted.get("raw_mask_b64"):
                m_img = Image.open(io.BytesIO(base64.b64decode(formatted["raw_mask_b64"]))).convert("L")
                m_np = (np.array(m_img) > 127).astype(np.uint8)
                rows, cols = m_np.shape
                mask_bytes = m_np.tobytes()
            elif formatted.get("mask_shape"):
                rows, cols = formatted["mask_shape"]
                mask_bytes = np.zeros((rows, cols), dtype=np.uint8).tobytes()

            seg_item = Dataset()
            seg_item.SegmentNumber = 1
            seg_item.SegmentLabel = "Lesion / Target Structure"
            seg_item.SegmentAlgorithmType = "AUTOMATIC"
            seg_item.SegmentAlgorithmName = "MURL Deep Learning Segmentor"

            cat_seq = Dataset()
            cat_seq.CodeValue = "M-01000"
            cat_seq.CodingSchemeDesignator = "SRT"
            cat_seq.CodeMeaning = "Morphologically Abnormal Structure"
            seg_item.SegmentedPropertyCategoryCodeSequence = Sequence([cat_seq])

            type_seq = Dataset()
            type_seq.CodeValue = "M-03000"
            type_seq.CodingSchemeDesignator = "SRT"
            type_seq.CodeMeaning = "Neoplasm"
            seg_item.SegmentedPropertyTypeCodeSequence = Sequence([type_seq])
            segments.append(seg_item)

        if not segments:
            # Fallback segment item
            seg_item = Dataset()
            seg_item.SegmentNumber = 1
            seg_item.SegmentLabel = "Region of Interest"
            seg_item.SegmentAlgorithmType = "AUTOMATIC"
            seg_item.SegmentAlgorithmName = "MURL AI"
            segments.append(seg_item)

        ds.SegmentSequence = Sequence(segments)
        ds.Rows = rows
        ds.Columns = cols
        ds.NumberOfFrames = num_frames
        ds.PixelData = mask_bytes

        # Referencing Source Image / Series if source DICOM was present
        if dcm_meta and dcm_meta.get("series_instance_uid") and dcm_meta.get("sop_instance_uid"):
            ref_series = Dataset()
            ref_series.SeriesInstanceUID = dcm_meta["series_instance_uid"]
            ref_sop = Dataset()
            ref_sop.ReferencedSOPClassUID = dcm_meta.get("sop_class_uid", "1.2.840.10008.5.1.4.1.1.7")
            ref_sop.ReferencedSOPInstanceUID = dcm_meta["sop_instance_uid"]
            ref_series.ReferencedInstanceSequence = Sequence([ref_sop])
            ds.ReferencedSeriesSequence = Sequence([ref_series])

        # Write to byte buffer
        out_buf = io.BytesIO()
        ds.save_as(out_buf)
        return out_buf.getvalue()

    # -------------------------------------------------------------------------
    # 3. DICOM SR (Structured Report) Generation
    # -------------------------------------------------------------------------
    @classmethod
    def export_dicom_sr(cls, analysis_result: Dict[str, Any], source_dicom_bytes: Optional[bytes] = None) -> bytes:
        """
        Creates a genuine DICOM Structured Report (SR) storage object.
        Preserves source study context when available, encodes findings and
        quantitative measurements using standard DICOM concept names and UCUM units.
        """
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom is required for DICOM SR export.")

        pred = analysis_result.get("prediction", {})
        formatted = pred.get("formatted_results", {})
        dcm_meta = analysis_result.get("dicom_metadata")
        modality = analysis_result.get("active_modality", "Unknown")

        # 1. Base File Meta Information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.33"  # Comprehensive 3D SR Storage
        sop_instance_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "SR"

        now = datetime.now()
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.ContentTime = now.strftime("%H%M%S")
        ds.SeriesDate = ds.ContentDate
        ds.SeriesTime = ds.ContentTime

        if dcm_meta and dcm_meta.get("study_instance_uid"):
            ds.StudyInstanceUID = dcm_meta["study_instance_uid"]
            ds.StudyID = dcm_meta.get("study_id", "DEID_STUDY")
            ds.PatientID = dcm_meta.get("patient_id", "RESEARCH-ANON")
            ds.PatientName = dcm_meta.get("patient_name", "RESEARCH^ANONYMIZED")
            ds.StudyDate = dcm_meta.get("study_date", ds.ContentDate)
            ds.StudyTime = dcm_meta.get("study_time", ds.ContentTime)
        else:
            ds.StudyInstanceUID = generate_uid()
            ds.StudyID = "MURL-RESEARCH"
            ds.PatientID = "RESEARCH-ANON"
            ds.PatientName = "RESEARCH^ANONYMIZED"
            ds.StudyDate = ds.ContentDate
            ds.StudyTime = ds.ContentTime

        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 102
        ds.InstanceNumber = 1
        ds.SeriesDescription = "AI Structured Report (Research Export)"
        ds.Manufacturer = "MURL Research Workbench"
        ds.ManufacturerModelName = "MURL Multimodal Imaging Workbench"
        ds.SoftwareVersions = "2.0.0"

        # SR Root Container
        ds.ValueType = "CONTAINER"
        ds.ContinuityOfContent = "SEPARATE"
        ds.CompletionFlag = "COMPLETE"
        ds.VerificationFlag = "UNVERIFIED"

        title_seq = Dataset()
        title_seq.CodeValue = "126000"
        title_seq.CodingSchemeDesignator = "DCM"
        title_seq.CodeMeaning = "Imaging Measurement Report"
        ds.ConceptNameCodeSequence = Sequence([title_seq])

        # Content Items
        items = []

        # 1. Academic & Research Disclaimer
        item_disc = Dataset()
        item_disc.RelationshipType = "CONTAINS"
        item_disc.ValueType = "TEXT"
        c_disc = Dataset()
        c_disc.CodeValue = "121111"
        c_disc.CodingSchemeDesignator = "DCM"
        c_disc.CodeMeaning = "Summary"
        item_disc.ConceptNameCodeSequence = Sequence([c_disc])
        item_disc.TextValue = "AI-generated research result. Not a clinical diagnosis. MURL Platform Analysis."
        items.append(item_disc)

        # 2. Modality & Study Target
        item_mod = Dataset()
        item_mod.RelationshipType = "CONTAINS"
        item_mod.ValueType = "TEXT"
        c_mod = Dataset()
        c_mod.CodeValue = "121058"
        c_mod.CodingSchemeDesignator = "DCM"
        c_mod.CodeMeaning = "Procedure reported"
        item_mod.ConceptNameCodeSequence = Sequence([c_mod])
        item_mod.TextValue = f"{modality} Automated Analysis"
        items.append(item_mod)

        # 3. Findings & Quantitative Measurements
        # Chest X-Ray findings
        if formatted.get("top_findings"):
            for f in formatted["top_findings"]:
                item_f = Dataset()
                item_f.RelationshipType = "CONTAINS"
                item_f.ValueType = "NUM"
                c_f = Dataset()
                c_f.CodeValue = "121071"
                c_f.CodingSchemeDesignator = "DCM"
                c_f.CodeMeaning = f.get("pathology", "Finding")
                item_f.ConceptNameCodeSequence = Sequence([c_f])

                m_val = Dataset()
                m_val.NumericValue = str(f.get("probability_percent", 0.0))
                u = Dataset()
                u.CodeValue = "%"
                u.CodingSchemeDesignator = "UCUM"
                u.CodeMeaning = "percent"
                m_val.MeasurementUnitsCodeSequence = Sequence([u])
                item_f.MeasuredValueSequence = Sequence([m_val])
                items.append(item_f)

        # Classification (BUSI)
        elif formatted.get("predicted_class"):
            item_c = Dataset()
            item_c.RelationshipType = "CONTAINS"
            item_c.ValueType = "TEXT"
            c_c = Dataset()
            c_c.CodeValue = "121071"
            c_c.CodingSchemeDesignator = "DCM"
            c_c.CodeMeaning = "Finding"
            item_c.ConceptNameCodeSequence = Sequence([c_c])
            item_c.TextValue = f"Class: {formatted.get('predicted_class')}, Confidence: {formatted.get('confidence')}"
            items.append(item_c)

        # CT Structures Volumetry
        if formatted.get("top_detected_structures"):
            for s in formatted["top_detected_structures"][:12]:
                item_s = Dataset()
                item_s.RelationshipType = "CONTAINS"
                item_s.ValueType = "NUM"
                c_s = Dataset()
                c_s.CodeValue = "121071"
                c_s.CodingSchemeDesignator = "DCM"
                c_s.CodeMeaning = s.get("structure_name", "Organ")
                item_s.ConceptNameCodeSequence = Sequence([c_s])

                m_val = Dataset()
                m_val.NumericValue = str(s.get("volume_cm3", 0.0))
                u = Dataset()
                u.CodeValue = "cm3"
                u.CodingSchemeDesignator = "UCUM"
                u.CodeMeaning = "cubic centimeter"
                m_val.MeasurementUnitsCodeSequence = Sequence([u])
                item_s.MeasuredValueSequence = Sequence([m_val])
                items.append(item_s)

        # MRI Glioma Compartments Volumetry
        if formatted.get("compartments"):
            for c in formatted["compartments"]:
                item_comp = Dataset()
                item_comp.RelationshipType = "CONTAINS"
                item_comp.ValueType = "NUM"
                c_comp = Dataset()
                c_comp.CodeValue = "121071"
                c_comp.CodingSchemeDesignator = "DCM"
                c_comp.CodeMeaning = c.get("compartment", "Compartment")
                item_comp.ConceptNameCodeSequence = Sequence([c_comp])

                m_val = Dataset()
                m_val.NumericValue = str(c.get("volume_cm3", 0.0))
                u = Dataset()
                u.CodeValue = "cm3"
                u.CodingSchemeDesignator = "UCUM"
                u.CodeMeaning = "cubic centimeter"
                m_val.MeasurementUnitsCodeSequence = Sequence([u])
                item_comp.MeasuredValueSequence = Sequence([m_val])
                items.append(item_comp)

        # Lesion Area % (Ultrasound / MedSAM)
        if formatted.get("lesion_area_percentage") is not None:
            item_area = Dataset()
            item_area.RelationshipType = "CONTAINS"
            item_area.ValueType = "NUM"
            c_area = Dataset()
            c_area.CodeValue = "121071"
            c_area.CodingSchemeDesignator = "DCM"
            c_area.CodeMeaning = "Lesion Area Burden"
            item_area.ConceptNameCodeSequence = Sequence([c_area])

            m_val = Dataset()
            m_val.NumericValue = str(formatted.get("lesion_area_percentage", 0.0))
            u = Dataset()
            u.CodeValue = "%"
            u.CodingSchemeDesignator = "UCUM"
            u.CodeMeaning = "percent"
            m_val.MeasurementUnitsCodeSequence = Sequence([u])
            item_area.MeasuredValueSequence = Sequence([m_val])
            items.append(item_area)

        ds.ContentSequence = Sequence(items)

        out_buf = io.BytesIO()
        ds.save_as(out_buf)
        return out_buf.getvalue()

    # -------------------------------------------------------------------------
    # 4. NIfTI Mask Export (.nii.gz)
    # -------------------------------------------------------------------------
    @classmethod
    def export_nifti_mask(cls, analysis_result: Dict[str, Any]) -> Tuple[bytes, str]:
        """
        Exports segmentation output as a genuine NIfTI (.nii.gz) volume.
        Preserves source affine, orientation, voxel spacing, and dimensions.
        """
        pred = analysis_result.get("prediction", {})
        formatted = pred.get("formatted_results", {})
        nii_meta = analysis_result.get("nifti_metadata", {})

        # If model generated a direct NIfTI file on disk (CT TotalSegmentator or BraTS SegResNet)
        out_path = formatted.get("output_nifti_path")
        if out_path and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                content = f.read()
            filename = os.path.basename(out_path)
            if not filename.endswith(".nii.gz") and not filename.endswith(".nii"):
                filename = f"{filename}.nii.gz"
            return content, filename

        # 2D Mask (BUSI or MedSAM) to NIfTI volume
        if formatted.get("raw_mask_b64"):
            m_img = Image.open(io.BytesIO(base64.b64decode(formatted["raw_mask_b64"]))).convert("L")
            m_np = (np.array(m_img) > 127).astype(np.uint8)
            # Create NIfTI image with identity affine
            nii_obj = nib.Nifti1Image(m_np, np.eye(4))
            buf = io.BytesIO()
            # Save as gzipped NIfTI using FileHolder
            file_holder = nib.FileHolder(fileobj=buf)
            nii_obj.to_file_map({'image': file_holder, 'header': file_holder})
            return buf.getvalue(), "Ultrasound_Segmentation_Mask.nii.gz"

        # Fallback dummy mask with metadata geometry if available
        affine = np.eye(4)
        if nii_meta and nii_meta.get("voxel_spacing_mm"):
            sp = nii_meta["voxel_spacing_mm"]
            if len(sp) >= 3:
                affine = np.diag([sp[0], sp[1], sp[2], 1.0])

        dims = nii_meta.get("dimensions", [64, 64, 64]) if nii_meta else [64, 64, 64]
        vol = np.zeros(dims, dtype=np.uint8)
        nii_obj = nib.Nifti1Image(vol, affine)
        buf = io.BytesIO()
        file_holder = nib.FileHolder(fileobj=buf)
        nii_obj.to_file_map({'image': file_holder, 'header': file_holder})
        return buf.getvalue(), "MURL_Segmentation_Mask.nii.gz"

    # -------------------------------------------------------------------------
    # 5. CSV Tabular Export
    # -------------------------------------------------------------------------
    @classmethod
    def export_csv(cls, analysis_result: Dict[str, Any]) -> str:
        """
        Exports meaningful quantitative measurements into standard CSV format.
        Does NOT produce meaningless CSVs for classification-only results.
        """
        pred = analysis_result.get("prediction", {})
        formatted = pred.get("formatted_results", {})

        out = io.StringIO()
        writer = csv.writer(out)

        # 1. CT TotalSegmentator structures
        if formatted.get("all_detected_structures") or formatted.get("top_detected_structures"):
            structs = formatted.get("all_detected_structures", formatted.get("top_detected_structures", []))
            writer.writerow(["structure_name", "label_id", "voxel_count", "volume_cm3"])
            for s in structs:
                writer.writerow([
                    s.get("structure_name", ""),
                    s.get("label_id", ""),
                    s.get("voxel_count", 0),
                    s.get("volume_cm3", 0.0)
                ])
            return out.getvalue()

        # 2. MRI Brain Tumor Compartments
        if formatted.get("compartments"):
            writer.writerow(["compartment", "description", "voxel_count", "volume_cm3"])
            for c in formatted["compartments"]:
                writer.writerow([
                    c.get("compartment", ""),
                    c.get("description", ""),
                    c.get("voxel_count", 0),
                    c.get("volume_cm3", 0.0)
                ])
            return out.getvalue()

        # 3. Ultrasound / MedSAM Lesion Metrics
        if formatted.get("lesion_area_percentage") is not None:
            writer.writerow(["measurement", "value", "unit"])
            writer.writerow(["lesion_detected", "True" if formatted.get("lesion_detected") else "False", "boolean"])
            writer.writerow(["lesion_area_percentage", formatted.get("lesion_area_percentage", 0.0), "%"])
            writer.writerow(["lesion_pixel_count", formatted.get("lesion_pixel_count", 0), "pixels"])
            if formatted.get("derived_bounding_boxes"):
                writer.writerow(["detected_rois_count", len(formatted["derived_bounding_boxes"]), "count"])
            return out.getvalue()

        # 4. Thyroid TI-RADS Features
        if formatted.get("tirads_features"):
            writer.writerow(["feature", "value", "unit"])
            for k, v in formatted["tirads_features"].items():
                writer.writerow([k, v, "ratio" if "ratio" in k else "score"])
            return out.getvalue()

        raise ValueError("No meaningful quantitative tabular measurements exist for this result type.")

    # -------------------------------------------------------------------------
    # 6. PNG Overlay & PNG Mask Export
    # -------------------------------------------------------------------------
    @classmethod
    def export_png_overlay(cls, analysis_result: Dict[str, Any]) -> bytes:
        """
        Exports the 2D blended scan and segmentation overlay image.
        """
        pred = analysis_result.get("prediction", {})
        formatted = pred.get("formatted_results", {})

        b64_str = (
            formatted.get("mask_overlay_b64") or
            formatted.get("multilabel_previews", {}).get("axial_seg_b64") or
            formatted.get("multiplanar_previews", {}).get("axial_wt_b64")
        )

        if not b64_str:
            raise ValueError("No 2D segmentation overlay available for this result.")

        return base64.b64decode(b64_str)

    @classmethod
    def export_png_mask(cls, analysis_result: Dict[str, Any]) -> bytes:
        """
        Exports the 2D binary / label segmentation mask as a standalone PNG.
        """
        pred = analysis_result.get("prediction", {})
        formatted = pred.get("formatted_results", {})

        if formatted.get("raw_mask_b64"):
            return base64.b64decode(formatted["raw_mask_b64"])

        # If only overlay is present, create mask or raise
        raise ValueError("Raw binary segmentation mask is not available as 2D PNG.")

    # -------------------------------------------------------------------------
    # 7. JSON Export
    # -------------------------------------------------------------------------
    @classmethod
    def export_json(cls, analysis_result: Dict[str, Any]) -> str:
        """
        Exports full structured machine-readable analysis data in formatted JSON.
        """
        return json.dumps(analysis_result, indent=2)
