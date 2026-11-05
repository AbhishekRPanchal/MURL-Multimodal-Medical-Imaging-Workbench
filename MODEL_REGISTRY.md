# MURL Multi-Modal Medical Imaging Platform — Model Registry Catalog

This document specifies the models registered in [`model_registry/registry.yaml`](model_registry/registry.yaml).
In accordance with platform safety policies and engineering standards, models are classified using a clear operational-status hierarchy.

---

## 1. Operational Status Taxonomy

To avoid ambiguity and prevent overclaiming, models are classified into six explicit engineering levels:

- **LEVEL 1 — Repository Available:** Upstream architecture/source code identified.
- **LEVEL 2 — Dependencies Available:** Python runtime environment requirements resolved.
- **LEVEL 3 — Adapter Implemented:** Thin platform wrapper with input preprocessing, format validation, and postprocessing integrated.
- **LEVEL 4 — Model Weights Available:** Pretrained or fine-tuned model checkpoint verified on disk.
- **LEVEL 5 — Inference Verified:** End-to-end inference execution verified through platform APIs on target hardware. *(Note: "Inference verified" means locally executable; it does NOT imply clinical validation).*
- **LEVEL 6 — Independent Evaluation:** Scientifically measured evaluation metrics documented on patient-disjoint held-out benchmark splits.

---

## 2. Registered Models Master Table

| Model ID | Name | Modality | Anatomy | Formats | Primary Task | Operational Status | Auto-Route |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `busi_resnet18_classifier` | BUSI Breast US ResNet18 | Ultrasound | Breast | png, jpg, dcm | 3-Class Classification | **Level 5 — Inference verified** (Level 6 test metrics) | Allowed |
| `busi_monai_unet_segmentor` | BUSI MONAI U-Net | Ultrasound | Breast | png, jpg, dcm | Lesion Boundary Segmentation | **Level 5 — Inference verified** (Level 6 test metrics) | Allowed |
| `totalsegmentator_ct` | TotalSegmentator CT | CT | Whole-Body | nii, nii.gz, dcm | 117-Organ Segmentation | **Level 5 — Inference verified** | Allowed |
| `chexpert_chest_xray_classifier` | TorchXRayVision DenseNet121 | X-ray | Chest | png, jpg, dcm, pdf | 18-Pathology Risk Scoring | **Level 5 — Inference verified** | Allowed |
| `monai_brats_mri_segmentor` | MONAI BraTS MRI | MRI | Brain | nii, nii.gz, dcm | MPR Viewer / SegResNet | **Partially Working** (MPR active; checkpoint optional) | Allowed (MPR) |
| `tn3k_thyroid_us_segmentor` | TN3K Thyroid US Analyzer | Ultrasound | Thyroid | png, jpg, dcm | Margin/Aspect Ratio Heuristic | **Partially Working** (Heuristic active; checkpoint optional) | Allowed (Heuristic) |
| `medsam_vit_b` | MedSAM ViT-B | Multi-Modal | Universal | png, jpg, dcm, nii | Prompt-Guided Segmentation | **Partially Working** (Box prompt active; checkpoint optional) | Manual Prompt |
| `nnunet_planner` | nnU-Net Planner | Multi-Modal | Universal | nii, nii.gz, dcm | Heuristic Dataset Fingerprinting | **Level 3 — Adapter implemented** | Manual |
| `monai_lung_nodule_ct_detector` | MONAI Lung Nodule CT | CT | Chest | nii, nii.gz, dcm | 3D Nodule Detection | **Level 1 — Planned** | Blocked |

---

## 3. Detailed Model Specifications

### 3.1 `busi_resnet18_classifier`
- **Name:** BUSI Breast Ultrasound ResNet18 Classifier
- **Modality:** Ultrasound
- **Anatomical Region:** Breast
- **Supported Formats:** `png`, `jpg`, `jpeg`, `dcm`, `dicom`
- **Input Dimensions:** `[256, 256]` (3 channels, ImageNet normalization)
- **Output Task:** 3-class Classification (`benign`, `malignant`, `normal`)
- **Dataset:** Breast Ultrasound Images Dataset (BUSI), Al-Dhabyani et al. (2020), CC BY 4.0
- **Held-Out Test Set Metrics (116 patient-disjoint cases):**
  - Accuracy: **84.48%** (98/116 correct)
  - Macro F1: **83.08%**
  - Macro ROC-AUC: **0.9483**
- **Status:** **Level 5 — Inference verified** (Level 6 evaluated on held-out test set)

### 3.2 `busi_monai_unet_segmentor`
- **Name:** BUSI Breast Ultrasound MONAI U-Net Segmentor
- **Modality:** Ultrasound
- **Anatomical Region:** Breast
- **Supported Formats:** `png`, `jpg`, `jpeg`, `dcm`, `dicom`
- **Input Dimensions:** `[256, 256]` (1 channel grayscale)
- **Output Task:** Lesion boundary segmentation & derived bounding-box extraction
- **Dataset:** BUSI (CC BY 4.0)
- **Held-Out Test Set Metrics (116 patient-disjoint cases):**
  - Mean Dice: **0.2623** (+98.0% over baseline 0.1325)
  - Mean IoU: **0.1652** (+111.5% over baseline 0.0781)
  - Malignant Sub-cohort Dice: **0.3684**
- **Status:** **Level 5 — Inference verified** (Level 6 evaluated on held-out test set)

### 3.3 `totalsegmentator_ct`
- **Name:** TotalSegmentator CT Anatomical Segmentor
- **Upstream Repository:** `https://github.com/wasserth/TotalSegmentator`
- **Modality:** CT (Computed Tomography)
- **Anatomical Region:** Whole-Body (117 anatomical structures)
- **Supported Formats:** `nii`, `nii.gz`, `dcm`, `dicom`
- **Input Dimensions:** 3D Volumetric NIfTI / DICOM series
- **Output Task:** Multi-organ segmentation and volumetric calculation (cm³)
- **Checkpoint:** Official upstream Task 297 (`Dataset297_TotalSegmentator_total_3mm_1559subj.zip`)
- **License:** Apache 2.0
- **Status:** **Level 5 — Inference verified**

### 3.4 `chexpert_chest_xray_classifier`
- **Name:** TorchXRayVision DenseNet121 Chest Radiograph Classifier
- **Upstream Repository:** `https://github.com/mlmed/torchxrayvision`
- **Modality:** X-ray (Chest Radiography)
- **Anatomical Region:** Chest / Thorax
- **Supported Formats:** `png`, `jpg`, `jpeg`, `dcm`, `dicom`, `pdf`
- **Input Dimensions:** `[224, 224]` (1 channel normalized)
- **Output Task:** 18-pathology risk profiling (Cardiomegaly, Effusion, Infiltration, etc.)
- **Checkpoint:** `densenet121-res224-all` (NIH + CheXpert + MIMIC-CXR)
- **License:** Apache 2.0
- **Status:** **Level 5 — Inference verified**

### 3.5 `monai_brats_mri_segmentor`
- **Name:** MONAI BraTS Multi-Planar Reconstruction & SegResNet Adapter
- **Upstream Repository:** `https://huggingface.co/MONAI/brats_mri_segmentation`
- **Modality:** MRI (Magnetic Resonance Imaging)
- **Anatomical Region:** Brain
- **Supported Formats:** `nii`, `nii.gz`, `dcm`, `dicom`
- **Capability:** Multi-Planar Reconstruction (MPR) across Axial, Coronal, Sagittal planes; SegResNet adapter ready for optional weights
- **Status:** **Partially Working** (MPR visualization active; neural checkpoint optional)

### 3.6 `tn3k_thyroid_us_segmentor`
- **Name:** TN3K Thyroid Ultrasound Heuristic & Adapter
- **Upstream Repository:** `https://github.com/geduer/TN3K`
- **Modality:** Ultrasound
- **Anatomical Region:** Thyroid
- **Supported Formats:** `png`, `jpg`, `jpeg`, `dcm`, `dicom`
- **Capability:** Aspect ratio (taller-than-wide) and margin irregularity acoustic analysis; adapter ready for TN3K weights
- **Status:** **Partially Working** (Heuristic analysis active; neural checkpoint optional)

### 3.7 `medsam_vit_b`
- **Name:** MedSAM Prompt-Guided Universal Medical Segmentor
- **Upstream Repository:** `https://github.com/bowang-lab/MedSAM`
- **Modality:** Multi-Modal
- **Supported Formats:** `png`, `jpg`, `jpeg`, `dcm`, `dicom`, `nii`, `nii.gz`
- **Capability:** Bounding-box prompt registration and pre-processing pipeline; ViT-B checkpoint optional
- **Status:** **Partially Working** (Prompt workflow active; neural checkpoint optional)
