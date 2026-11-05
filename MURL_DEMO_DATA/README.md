# MURL Medical Imaging Demo & Verification Dataset

**PURPOSE:** DEMO / MANUAL TESTING / UI VERIFICATION ONLY.  
*Included for research/demo purposes; users must verify the original dataset/model terms before redistribution.*
*Not for model training, fine-tuning, scientific benchmarking, clinical validation, or unverified dataset redistribution.*

**Date Prepared / Downloaded:** 2026-11-05

---

## Sample Index

| Modality | Folder | File Name | Size | Source / Dataset Name | License | Status |
|:---|:---|:---|---:|:---|:---|:---|
| **Breast Ultrasound** | `breast_ultrasound/` | `busi_benign_01.png` | 326 KB | BUSI (Al-Dhabyani et al., 2020) | CC BY 4.0 | Level 5 — Inference verified |
| **Breast Ultrasound** | `breast_ultrasound/` | `busi_malignant_01.png` | 300 KB | BUSI (Al-Dhabyani et al., 2020) | CC BY 4.0 | Level 5 — Inference verified |
| **Breast Ultrasound** | `breast_ultrasound/` | `busi_normal_01.png` | 278 KB | BUSI (Al-Dhabyani et al., 2020) | CC BY 4.0 | Level 5 — Inference verified |
| **Chest X-ray** | `chest_xray/` | `nih_cxr_cardiomegaly_01.png` | 178 KB | NIH ChestX-ray14 / Open-I | Public Domain | Level 5 — Inference verified |
| **Chest X-ray** | `chest_xray/` | `cxr_normal_02.png` | 24 KB | NIH ChestX-ray14 / Open-I | Public Domain | Level 5 — Inference verified |
| **Brain MRI** | `brain_mri/` | `brats_brain_mri_demo.nii.gz` | 454 KB | BraTS Challenge Benchmark | Apache 2.0 | Partially Working (MPR active; checkpoint required) |
| **Thyroid Ultrasound**| `thyroid_ultrasound/` | `tn3k_thyroid_sample_0000.jpg` | 38 KB | TN3K (Gong et al., 2023) | CC BY-NC-SA 4.0 | Partially Working (Heuristic analysis active; checkpoint required) |
| **Thyroid Ultrasound**| `thyroid_ultrasound/` | `tn3k_thyroid_sample_0001.jpg` | 110 KB | TN3K (Gong et al., 2023) | CC BY-NC-SA 4.0 | Partially Working (Heuristic analysis active; checkpoint required) |
| **Computed Tomography**| `ct/` | `example_ct_sm.nii.gz` | 472 KB | TotalSegmentator (Wasserthal et al., 2023) | Apache 2.0 | Level 5 — Inference verified |
| **DICOM** | `dicom/` | `CTHead1.dcm` | 9.4 KB | 3D Slicer Testing Suite | BSD-style | Level 5 — Inference verified |
| **Multi-page PDF** | `pdf/` | `multipage_medical_scans.pdf` | 78 KB | Synthetic multipage test document (X-ray + Ultrasound) | MURL Research Suite | Level 5 — Inference verified |
| **MedSAM (Universal)**| `breast_ultrasound/` | Reuses `busi_benign_01.png` | — | Reused BUSI ultrasound raster | CC BY 4.0 | Partially Working (Box prompt active; checkpoint required) |

*Note on CT Data: `example_ct_sm.nii.gz` is the original CT input scan. Any `totalseg_out_*.nii.gz` files generated during inference are segmentation label masks and must never be treated as raw CT acquisition inputs.*
