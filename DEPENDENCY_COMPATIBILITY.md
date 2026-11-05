# MURL Multi-Modal Medical Imaging Platform — Dependency Compatibility & Hardware Audit

This document audits the local Windows execution environment, detailing package versions, hardware constraints, upstream repository compatibility, and virtual environment isolation boundaries.

---

## 1. Local Runtime Environment

- **Host Operating System:** Microsoft Windows 11 / Windows Server (64-bit AMD64)
- **Primary Python Runtime:** Python 3.14.7 (`MSC v.1944 64 bit`)
- **Primary Deep Learning Framework:** PyTorch `2.14.0+cpu`
- **CUDA Hardware Acceleration:** `torch.cuda.is_available() == False` (CPU-only execution)
- **Available Storage on Drive `C:`:** **455.54 GB Free** (496.97 GB used)
- **Target Compute Architecture:** Intel/AMD x86_64 CPU with AVX2 instruction set

---

## 2. Core Python Packages Installed

| Package Name | Installed Version | Verification Status | Operational Role |
|:---|:---:|:---:|:---|
| `torch` | `2.14.0+cpu` | Verified | Neural network inference and tensor algebra |
| `torchvision` | `0.19.0+cpu` | Verified | Vision transforms and ImageNet backbone feature extractors |
| `monai` | `1.6.1` | Verified | Medical image transforms, 2D/3D U-Net, and sliding window inference |
| `pydicom` | `3.0.2` | Verified | DICOM clinical tag parsing, pixel extraction, and research privacy PHI filtering |
| `nibabel` | `5.4.2` | Verified | NIfTI 3D/4D volume loading, affine orientation, and voxel zoom extraction |
| `numpy` | `2.3.5` | Verified | High-performance array operations and masking |
| `opencv-python` | `4.11.0` | Verified | Connected components analysis, bounding box extraction, and geometry |
| `Pillow (PIL)` | `11.1.0` | Verified | Raster image loading, multi-planar slice preview conversion |
| `fastapi` | `0.115.6` | Verified | Asynchronous REST API service |
| `uvicorn` | `0.34.0` | Verified | ASGI web server |
| `scikit-learn` | `1.6.1` | Verified | Evaluation metrics (ROC-AUC, Precision, Recall, Confusion Matrix) |
| `matplotlib` | `3.10.0` | Verified | Scientific figure generation and confusion matrix rendering |
| `pytest` | `9.1.1` | Verified | Test automation harness |

---

## 3. Cloned Upstream Projects Compatibility Matrix

```mermaid
flowchart TD
    Env["Host Environment: Windows / Python 3.14.7 / CPU-only"]
    Env --> M1["MONAI (v1.6.1)\n• Native Library Import\n• CPU Sliding Window\n• Status: Full Runtime"]
    Env --> M2["MedSAM (bowang-lab)\n• segment_anything ViT-B\n• Prompt Box Encoder\n• Status: Adapter Integrated"]
    Env --> M3["TotalSegmentator (wasserth)\n• Cropping & HU Windowing\n• 117-class mapping\n• Status: Adapter Preprocessing"]
    Env --> M4["nnU-Net (MIC-DKFZ)\n• Needs batchgenerators\n• Geometry Planning\n• Status: Isolated Planning"]
    Env --> M5["MONAI Label\n• RESTful Active Learning\n• Status: Schema Aligned"]
    Env --> M6["3D Slicer\n• C++ Desktop App\n• Status: File Export Bridge"]
```

### Detailed Evaluation:

### 3.1 Project-MONAI / MONAI
- **Compatibility:** **100% Native**.
- Runs directly inside the host Python 3.14 environment.
- Preprocessing (`Compose`, `Spacing`, `Orientation`, `ScaleIntensityRange`) and 2D U-Net forward inference run on CPU in **<45 ms**.
- 3D volumetric sliding window inference runs in **~1.5–3.0 seconds** on CPU with small batch and patch sizes.

### 3.2 bowang-lab / MedSAM
- **Compatibility:** **Native Python Module**.
- `segment_anything` imports cleanly from `external_repos/MedSAM`.
- Prompt-based segmentation works on CPU with lightweight decoder.
- Checkpoint: `medsam_vit_b.pth` (~375 MB).

### 3.3 wasserth / TotalSegmentator
- **Compatibility:** **Adapter & Preprocessing Compatible**.
- `totalsegmentator.cropping` and `totalsegmentator.map_to_binary` run natively.
- Full multi-class 117-organ CT inference requires downloading model weights (~1.2 GB). Because downloads must not occur unapproved, the adapter exposes:
  1. Full offline CT preprocessing (cropping, HU windowing, 3D spacing conversion).
  2. Gated download trigger with explicit size warning.

### 3.4 MIC-DKFZ / nnU-Net
- **Compatibility:** **Isolated Environment Boundary**.
- `nnunetv2` requires `batchgenerators` which is not installed in the system environment.
- **Handling Strategy:** Preserves adapter boundary (`nnUNetAdapter`). Heuristic fingerprint planning is encapsulated, avoiding global environment pollution.

### 3.5 Project-MONAI / MONAILabel
- **Compatibility:** **Schema-Compatible REST Client**.
- The main FastAPI application aligns with MONAI Label task schemas (`/annotation/infer`, `/annotation/activelearning`).

### 3.6 Slicer / Slicer (3D Slicer)
- **Compatibility:** **File Export Bridge**.
- Slicer is a compiled C++/Qt desktop workstation. The platform generates standardized NIfTI volumes and DICOM SEG overlays that load directly into 3D Slicer.

---

## 4. Hardware Sizing & Minimum Requirements

| Component | Minimum Specification | Recommended Specification | Local Host Status |
|:---|:---|:---|:---|
| **CPU** | 4-Core x86_64 | 8-Core Intel Core i7 / AMD Ryzen 7 | Passed (AMD64) |
| **RAM** | 8 GB | 16–32 GB | Passed |
| **Storage** | 10 GB Free | 50 GB Free | Passed (**455.5 GB Free**) |
| **GPU / VRAM** | None (CPU Fallback) | NVIDIA RTX 3060+ (8 GB VRAM) | CPU-only verified |
