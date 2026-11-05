# MURL Multi-Modal Medical Imaging Platform — Multi-Modal Input Formats & Processing Specification

This document details the multi-modal input ingestion, validation, and metadata extraction pipelines supporting **PNG, JPG/JPEG, DICOM, and NIfTI (`.nii`, `.nii.gz`)** in the MURL platform.

---

## 1. Supported Formats and Ingestion Matrix

| Format | Extension | Underlying Engine | Dimensions Handled | Native Technical Metadata | Typical Modalities |
|:---|:---|:---|:---:|:---|:---|
| **Raster (Standard)** | `.png`, `.jpg`, `.jpeg` | PIL (Pillow), OpenCV | 2D | Color space, pixel resolution, aspect ratio | Ultrasound, Dermatology, Standard Photography |
| **DICOM (Clinical)** | `.dcm`, `.dicom` | `pydicom` | 2D, Multi-frame 3D | Modality tag, BodyPartExamined, Spacing, Slice Thickness, Windowing, Photometric Int. | Ultrasound, CT, MRI, Radiography (X-ray) |
| **NIfTI (Research)** | `.nii`, `.nii.gz` | `nibabel`, `monai` | 3D Volumetric, 4D Multi-parametric | Affine matrix, Voxel spacing (zooms in mm), Anatomical orientation (RAS/LPS), Units | Brain MRI, Thoracic/Abdominal CT, Functional MRI |

---

## 2. Ingestion Pipeline Stages

```mermaid
flowchart TD
    File["Incoming File Stream"] --> Gate["Stage 1: ImageFormatValidator\n(Extension, Magic Header, Size Limits)"]
    Gate -->|Rejected| Err["HTTP 400 Format Error"]
    Gate -->|Accepted| RouteFormat{"Stage 2: Format Dispatcher"}
    RouteFormat -->|DICOM| DicomHandler["DICOMMetadataParser\n• Research Privacy PHI Filtering\n• Rescale Slope/Intercept\n• Windowing Calculation"]
    RouteFormat -->|NIfTI| NiftiHandler["NIfTIMetadataParser\n• Affine & Orientation (RAS)\n• Voxel Zoom Extraction (mm)\n• Multi-Planar Orthogonal Slicing"]
    RouteFormat -->|Raster (PNG/JPG)| RasterHandler["PIL Image Normalizer\n• Aspect Ratio\n• Dynamic Range Check"]
    DicomHandler --> Modality["Stage 3: ModalityDetector\n• Header Confidence (1.0)\n• Raster Acoustic/Speckle Heuristic\n• Uncertainty & Confirmation Flag"]
    NiftiHandler --> Modality
    RasterHandler --> Modality
    Modality --> Compat["Stage 4: ModelCompatibilityValidator\n• Modality alignment\n• Spatial dimensions (2D vs 3D)\n• Channels & MRI sequences"]
```

---

## 3. Format Validation and Security Controls

### 3.1 Binary Magic Header Validation
To prevent extension-spoofing attacks or corrupted inputs, `ImageFormatValidator` checks the initial bytes:
- **PNG:** `\x89PNG\r\n\x1a\n` (Bytes 0–7)
- **JPEG:** `\xff\xd8\xff` (Bytes 0–2)
- **DICOM:** `DICM` ASCII marker located at byte offset 128 (following the 128-byte preamble) or group tags `0x0002` / `0x0008`
- **NIfTI (`.nii.gz`):** Gzip header `\x1f\x8b` (Bytes 0–1)
- **NIfTI (`.nii` uncompressed):** Header size `sizeof_hdr == 348` (0x0000015C) at offset 0

### 3.2 File Size Constraints
- **2D Images (PNG, JPG, Single-frame DICOM):** Capped at **50 MB**.
- **3D / 4D Volumes (NIfTI, Multi-frame DICOM):** Capped at **250 MB**.
Files exceeding these boundaries are rejected before loading into memory to protect against memory exhaustion on CPU.

### 3.3 Secure Temporary File Management
When parsing file streams (especially compressed `.nii.gz` volumes that require seekable file descriptors), files are stored using `tempfile.NamedTemporaryFile` with isolated UUID paths and guaranteed deletion in `finally:` blocks.

---

## 4. DICOM Metadata Extraction and Research Privacy PHI Filtering

Clinical DICOM files contain rich acquisition parameters alongside sensitive Protected Health Information (PHI).

### 4.1 Extracted Technical Metadata
- **Modality tag `(0008, 0060)`:** Standard code (`US`, `CT`, `MR`, `DX`, `CR`, `MG`)
- **Body Part Examined `(0018, 0015)`:** E.g., `BREAST`, `CHEST`, `HEAD`
- **Image Dimensions:** `Rows`, `Columns`, `NumberOfFrames`
- **Pixel Spacing `(0028, 0030)`:** Physical distance in mm between adjacent pixel centers `[row_spacing, col_spacing]`
- **Slice Thickness `(0018, 0050)`:** Nominal slice thickness in mm
- **Windowing `(0028, 1050 / 1051)`:** Window Center and Window Width for CT contrast visualization
- **Rescale Slope & Intercept `(0028, 1052 / 1053)`:** Converts raw stored values to calibrated Hounsfield Units (HU) via:
  $$\text{HU} = (\text{PixelValue} \times \text{RescaleSlope}) + \text{RescaleIntercept}$$

### 4.2 Research Privacy Filtering (De-Identification)
Before metadata leaves the parser, the following 16 PHI tags are purged or redacted:
- `PatientName`, `PatientID`, `PatientBirthDate`, `PatientSex`, `PatientAge`, `PatientAddress`
- `InstitutionName`, `InstitutionAddress`, `ReferringPhysicianName`, `PerformingPhysicianName`
- `OperatorsName`, `AccessionNumber`, `StudyID`, `OtherPatientIDs`, `EthnicGroup`, `MedicalRecordLocator`

---

## 5. Volumetric NIfTI 3D & 4D Processing

NIfTI volumes are inspected using `nibabel` and `monai`:
- **Voxel Spacing (zooms):** Physical dimensions of each voxel in mm $(dx, dy, dz)$.
- **Anatomical Orientation:** Determined via `nibabel.aff2axcodes(img.affine)` to map scan axes to human anatomy (e.g. `['R', 'A', 'S']` for Right-Anterior-Superior).
- **4D Time-Series / Multi-Parametric MRI:** Correctly separates the 4th dimension (e.g., T1, T1ce, T2, FLAIR sequences or dynamic contrast acquisitions).
- **Multi-Planar Orthogonal Reconstruction (MPR):**
  - **Axial View:** Slice along the superior-inferior axis $(z)$.
  - **Coronal View:** Slice along the anterior-posterior axis $(y)$.
  - **Sagittal View:** Slice along the left-right axis $(x)$.
  The mid-slices are extracted, percentile-windowed (1st to 99th percentile), and rendered to 8-bit previews for real-time frontend visualization.

---

## 6. Modality Uncertainty and Ethical Rules

1. **DICOM Priority:** When DICOM header metadata is available, modality is verified with 100% confidence and zero uncertainty.
2. **Raster Estimation:** For PNG/JPG, acoustic speckle, border dark margin ratio, and dynamic range are evaluated to produce a probabilistic estimate (`predicted_modality`, `probability`, `uncertainty`).
3. **User Confirmation Required:** Because raster files lack clinical metadata, `needs_user_confirmation: True` is always asserted, providing a confirmation dialog in the user interface.
4. **Strict Ethical Ban:** The system never infers anatomical diagnosis or findings from filenames.
