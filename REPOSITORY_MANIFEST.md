# MURL Multi-Modal Medical Imaging Platform — Repository Manifest

This manifest documents the external open-source medical AI repositories referenced and integrated by the platform.
All repositories were cloned from their official distributions using shallow depth (`--depth 1`) to preserve workspace storage while capturing the authoritative upstream source code.

---

## 1. Cloned Repositories Summary Table

| Repository | Official Upstream URL | Cloned Commit Hash | Upstream License | Primary Reusable Capabilities | Integration Strategy |
|:---|:---|:---:|:---:|:---|:---|
| **MONAI** | `https://github.com/Project-MONAI/MONAI` | `0e9fc83b19352fd624ce99d861e1626c6076fbaf` | Apache 2.0 | Medical transforms, 2D/3D U-Net & DynUNet backbones, sliding window inference, evaluation metrics (Dice, Hausdorff) | Direct Python library import (`monai`) + Model Zoo architecture integration |
| **nnU-Net** | `https://github.com/MIC-DKFZ/nnUNet` | `517ddab5794f5c4e9083d4341fd99cdcb4d7c1ce` | Apache 2.0 | Automated self-configuring preprocessing, fingerprint extraction, resample heuristics, patch-based inference | Preprocessing planning engine & architecture heuristics adapter |
| **TotalSegmentator** | `https://github.com/wasserth/TotalSegmentator` | `c59be95b839252dd993e1e32ff3369168d9f4eab` | Apache 2.0 | 117-class anatomical CT segmentation, body part crop heuristics, Hounsfield Unit (HU) windowing | Thin Python adapter wrapping CT preprocessing, sub-sampling, and model router |
| **MedSAM** | `https://github.com/bowang-lab/MedSAM` | `d71e8a1a99ad751840a22a7fa3ecfb4166fb1488` | Apache 2.0 | Universal medical promptable segmentation, bounding-box to mask propagation, Vision Transformer (ViT) feature extraction | Prompt-based segmentation adapter supporting derived/user bounding boxes |
| **MONAI Label** | `https://github.com/Project-MONAI/MONAILabel` | `c6c7b46670452c03d7c1ff166f6e10dcfdc2d99d` | Apache 2.0 | Active learning, human-in-the-loop interactive annotation, RESTful annotation API, DICOMweb integration | Annotation interface specification & active learning schema |
| **3D Slicer** | `https://github.com/Slicer/Slicer` | `a68359e86d16865f3f34ffd428b36434bed1c114` | BSD-style Slicer Agreement | 2D/3D volumetric visualization, multi-planar reconstruction (MPR), DICOM database management | Standalone visualization & OpenIGTLink/SlicerWeb export bridge |

---

## 2. Detailed Upstream Repository Audits

### 2.1 Project-MONAI / MONAI
- **Path:** `external_repos/MONAI`
- **Upstream Commit:** `0e9fc83b19352fd624ce99d861e1626c6076fbaf`
- **License:** Apache License 2.0 (Permissive open source)
- **Reusable Modules Inspected:**
  - `monai.transforms`: `Compose`, `LoadImage`, `EnsureChannelFirst`, `ScaleIntensityRange`, `Spacing`, `Orientation`, `ResizeWithPadOrCrop`
  - `monai.networks.nets`: `UNet`, `DynUNet`, `VNet`, `HighResNet`, `DenseNet121`
  - `monai.inferers`: `sliding_window_inference`, `SimpleInferer`
  - `monai.metrics`: `DiceMetric`, `HausdorffDistanceMetric`, `SurfaceDistanceMetric`
  - `monai.data`: `CacheDataset`, `DataLoader`, `write_nifti`
- **System Role:** Core foundational medical deep learning engine for the platform.

### 2.2 MIC-DKFZ / nnU-Net
- **Path:** `external_repos/nnUNet`
- **Upstream Commit:** `517ddab5794f5c4e9083d4341fd99cdcb4d7c1ce`
- **License:** Apache License 2.0
- **Reusable Modules Inspected:**
  - `nnunetv2.preprocessing.preprocessors`: `DefaultPreprocessor`, resampling algorithms preserving voxel geometry
  - `nnunetv2.experiment_planning`: Dataset fingerprint extraction (spacing percentiles, foreground intensity distributions)
  - `nnunetv2.inference.predict_from_raw_data`: Sliding window patch inference with Gaussian weighting
- **System Role:** Gold-standard automated heuristic planner and multi-modal preprocessing benchmark.

### 2.3 wasserth / TotalSegmentator
- **Path:** `external_repos/TotalSegmentator`
- **Upstream Commit:** `c59be95b839252dd993e1e32ff3369168d9f4eab`
- **License:** Apache License 2.0
- **Reusable Modules Inspected:**
  - `totalsegmentator.bin.TotalSegmentator`: Main CLI entry point
  - `totalsegmentator.python_api`: `totalsegmentator(input, output, task='total', ...)`
  - `totalsegmentator.map_to_binary`: Organ-specific label map mappings (117 anatomical structures)
  - `totalsegmentator.cropping`: Fast body-part and organ bounding-box localization
- **System Role:** Dedicated CT anatomical segmentation and anatomical ROI extraction.

### 2.4 bowang-lab / MedSAM
- **Path:** `external_repos/MedSAM`
- **Upstream Commit:** `d71e8a1a99ad751840a22a7fa3ecfb4166fb1488`
- **License:** Apache License 2.0
- **Reusable Modules Inspected:**
  - `segment_anything`: Meta SAM architecture foundation
  - `medsam_inference.py`: Preprocessing 1024x1024 embedding calculation and prompt box encoding
  - Forward decoder: Bounding box prompt $\to$ dense lesion/organ mask
- **System Role:** Interactive and prompt-guided multi-modal segmentation (e.g. converting derived ultrasound/CT bounding boxes to refined segmentation boundaries).

### 2.5 Project-MONAI / MONAILabel
- **Path:** `external_repos/MONAILabel`
- **Upstream Commit:** `c6c7b46670452c03d7c1ff166f6e10dcfdc2d99d`
- **License:** Apache License 2.0
- **Reusable Modules Inspected:**
  - `monailabel.endpoints`: REST API specification for active learning and model serving
  - `monailabel.interfaces.app`: `MONAILabelApp` lifecycle and task registration
  - `monailabel.interfaces.tasks`: `InferTask`, `TrainTask`, `ScoringMethod`
- **System Role:** Blueprint for standardized research annotations and human-in-the-loop active learning workflows.

### 2.6 Slicer / Slicer (3D Slicer)
- **Path:** `external_repos/Slicer`
- **Upstream Commit:** `a68359e86d16865f3f34ffd428b36434bed1c114`
- **License:** Custom BSD-style 3D Slicer Software License Agreement
- **Reusable Capabilities Inspected:**
  - `Modules/CLI`: Command line module specifications for external pipelines
  - `Base/QTGUI`: UI paradigms for slice viewers and 3D surface rendering
  - Medical image display conventions: Multi-Planar Reconstruction (Axial, Coronal, Sagittal)
- **System Role:** External visualization standard and reference for multi-planar medical image inspection.

---

## 3. Storage and Checkpoint Management Safeguards

1. No unvetted model checkpoints are stored directly in the git repository.
2. Large binary weight downloads are handled via dedicated scripts with destination validation.
3. Upstream libraries and dependencies are referenced cleanly via `requirements.txt` and verified adapters.
