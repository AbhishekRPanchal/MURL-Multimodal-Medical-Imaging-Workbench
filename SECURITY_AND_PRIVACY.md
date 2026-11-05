# MURL Multi-Modal Medical Imaging Platform — Security, Privacy & Safety Protocols

This document specifies the privacy engineering, security gates, and research safety controls implemented in the MURL platform for **Research Privacy Filtering** and modern cybersecurity best practices. Note: MURL is a research workbench and does not claim certified HIPAA compliance.

---

## 1. Research Privacy Filtering & De-Identification Protocol

For medical imaging research, systems must strip individual patient identifiers to ensure research de-identification.

### 1.1 Automated PHI Stripping in DICOM Ingestion
When a DICOM file is ingested by [`DICOMMetadataParser`](src/platform/dicom_parser.py), all metadata is parsed through a strict whitelist and sanitization pipeline.
The following **16 Protected Health Information (PHI) DICOM elements** are explicitly scrubbed and removed prior to returning API responses or logging:

| DICOM Tag | Keyword | Sanitization Action |
|:---:|:---|:---|
| `(0010, 0010)` | `PatientName` | **Purged** (Never exposed to API or UI) |
| `(0010, 0020)` | `PatientID` | **Purged** (Substituted with session UUID) |
| `(0010, 0030)` | `PatientBirthDate` | **Purged** (Removed) |
| `(0010, 0040)` | `PatientSex` | **Purged** (Removed) |
| `(0010, 1010)` | `PatientAge` | **Purged** (Removed) |
| `(0010, 1000)` | `OtherPatientIDs` | **Purged** (Removed) |
| `(0010, 1040)` | `PatientAddress` | **Purged** (Removed) |
| `(0010, 2160)` | `EthnicGroup` | **Purged** (Removed) |
| `(0008, 0080)` | `InstitutionName` | **Redacted** (Masked as `"ANONYMIZED"`) |
| `(0008, 0081)` | `InstitutionAddress` | **Purged** (Removed) |
| `(0008, 0090)` | `ReferringPhysicianName` | **Purged** (Removed) |
| `(0008, 1050)` | `PerformingPhysicianName` | **Purged** (Removed) |
| `(0008, 1070)` | `OperatorsName` | **Purged** (Removed) |
| `(0008, 0050)` | `AccessionNumber` | **Purged** (Removed) |
| `(0020, 0010)` | `StudyID` | **Purged** (Removed) |
| `(0010, 1090)` | `MedicalRecordLocator` | **Purged** (Removed) |

### 1.2 Preservation of Non-Identifying Clinical Physics Data
While PHI is purged, critical imaging acquisition parameters are preserved to enable downstream AI inference:
- `Modality` (e.g. `US`, `CT`, `MR`, `DX`)
- `BodyPartExamined` (e.g. `BREAST`, `CHEST`)
- `PixelSpacing` $[dx, dy]$ and `SliceThickness` in millimeters
- `Rows` and `Columns`
- `WindowCenter` and `WindowWidth`
- `RescaleSlope` and `RescaleIntercept`

---

## 2. Input Validation and System Hardening

### 2.1 Binary Magic Header Verification
File extensions can be forged or renamed. To prevent injection attacks and runtime crashes, [`ImageFormatValidator`](src/platform/format_validator.py) checks the underlying magic byte header:
- **PNG:** `\x89PNG\r\n\x1a\n`
- **JPEG:** `\xff\xd8\xff`
- **DICOM:** `DICM` ASCII sequence at byte offset 128 (or standard group tags `0x0002` / `0x0008`)
- **NIfTI (Gzip):** `\x1f\x8b`
- **NIfTI (Uncompressed):** Header byte length `sizeof_hdr == 348`

If a file has a `.png` extension but contains executable binaries or mismatched byte streams, it is immediately rejected with an HTTP 400 error.

### 2.2 DoS and Decompression Bomb Mitigation
- **2D Image Upload Limit:** Capped at **50 MB**.
- **3D/4D Volumetric Limit:** Capped at **250 MB**.
- Large image allocations exceeding these bounds are blocked prior to memory decompression.

### 2.3 Ephemeral File Isolation
For volumetric formats requiring seekable file handles (`.nii.gz`), temporary files are allocated using `tempfile.NamedTemporaryFile` with random filenames in the OS temporary directory and are deleted in `finally:` blocks immediately after metadata extraction.

---

## 3. Ethical Clinical AI Boundaries

1. **Strict Prohibition on Filename Inference:** The system never infers patient pathology, anatomical region, or diagnosis from filenames (e.g., `cancer_patient.png`). Modality is inferred solely from verified DICOM tags or image-based acoustic heuristics with explicit uncertainty reporting.
2. **Mandatory User Confirmation for Rasters:** Standard PNG/JPG scans lack DICOM acquisition provenance. The system flags `needs_user_confirmation: True` and prompts the user to confirm the target modality before routing.
3. **Safety Gate on Unvalidated Models:** Automated routing is restricted to models with `validation_status: "validated"`. Planned or experimental models cannot be triggered without explicit manual selection and parameter verification.
4. **Research Use Notice:** Every API payload, report, and frontend card includes the regulatory notice:
   > *"This platform is intended for research and educational purposes. Outputs are experimental and are not a medical diagnosis or a substitute for qualified clinical interpretation."*
