# MURL Multi-Modal Medical Imaging Platform — License & Attribution Notices

This document details the software licenses, copyright notices, and academic citations for all external open-source projects, frameworks, and datasets integrated into the MURL platform.

---

## 1. Cloned Open-Source Repositories

### 1.1 Project-MONAI / MONAI
- **License:** Apache License, Version 2.0
- **Copyright:** (c) 2020 - 2026 MONAI Consortium
- **Upstream Repository:** `https://github.com/Project-MONAI/MONAI`
- **Citation:**
  > The MONAI Consortium. (2020). *MONAI: Medical Open Network for AI*. Zenodo. doi:10.5281/zenodo.4323059.

### 1.2 MIC-DKFZ / nnU-Net
- **License:** Apache License, Version 2.0
- **Copyright:** (c) German Cancer Research Center (DKFZ) and Applied Computer Vision Lab (ACVL)
- **Upstream Repository:** `https://github.com/MIC-DKFZ/nnUNet`
- **Citation:**
  > Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods, 18(2), 203-211.

### 1.3 wasserth / TotalSegmentator
- **License:** Apache License, Version 2.0
- **Copyright:** (c) Jakob Wasserthal, University Hospital Basel
- **Upstream Repository:** `https://github.com/wasserth/TotalSegmentator`
- **Citation:**
  > Wasserthal, J., Breit, H. C., Meyer, M. T., Pradella, M., Hinck, D., Sauter, A. W., ... & Yang, S. (2023). *TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images*. Radiology: Artificial Intelligence, 5(5), e230024.

### 1.4 bowang-lab / MedSAM
- **License:** Apache License, Version 2.0
- **Copyright:** (c) Bo Wang Lab, University of Toronto & Vector Institute
- **Upstream Repository:** `https://github.com/bowang-lab/MedSAM`
- **Citation:**
  > Ma, J., He, Y., Li, F., Han, L., You, C., & Wang, B. (2024). *Segment anything in medical images*. Nature Communications, 15(1), 654.

### 1.5 Project-MONAI / MONAILabel
- **License:** Apache License, Version 2.0
- **Copyright:** (c) 2020 - 2026 MONAI Consortium
- **Upstream Repository:** `https://github.com/Project-MONAI/MONAILabel`
- **Citation:**
  > Diaz-Pinto, A., Alle, S., Amrulloh, Y., et al. (2024). *MONAI Label: A framework for AI-assisted interactive labeling of 3D medical images*. arXiv:2203.12362.

### 1.6 Slicer / Slicer (3D Slicer)
- **License:** 3D Slicer Contribution and Software License Agreement (BSD-style)
- **Copyright:** (c) The Brigham and Women's Hospital, Inc. and Slicer Community
- **Upstream Repository:** `https://github.com/Slicer/Slicer`
- **Citation:**
  > Fedorov, A., Beichel, R., Kalpathy-Cramer, J., Finet, J., Fillion-Robin, J. C., Pujol, S., ... & Kikinis, R. (2012). *3D Slicer as an image computing platform for the Quantitative Imaging Network*. Magnetic Resonance Imaging, 30(9), 1323-1341.

---

## 2. Integrated & Demo Datasets

### 2.1 Breast Ultrasound Images Dataset (BUSI)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Citation:**
  > Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). *Dataset of breast ultrasound images*. Data in Brief, 28, 104863.

### 2.2 NIH ChestX-ray14 / Open-I Radiographs
- **License:** Public Domain / Open-I Open Access Biomedical Image Search
- **Citation:**
  > Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*. IEEE CVPR.

### 2.3 TotalSegmentator CT Demo Volume
- **License:** Apache License, Version 2.0
- **Source:** Jakob Wasserthal, University Hospital Basel. TotalSegmentator v2.0.0.

### 2.4 BraTS Brain MRI Benchmark Volume
- **License:** Apache License, Version 2.0 (MONAI Model Zoo BraTS)
- **Citation:**
  > Baid, U., et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification*. arXiv:2107.02314.

### 2.5 3D Slicer CT Sample (`CTHead1.dcm`)
- **License:** 3D Slicer BSD-style license
- **Source:** Slicer Community / Brigham and Women's Hospital.

### 2.6 TN3K Thyroid Ultrasound Dataset
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Citation:**
  > Gong, H., et al. (2023). *TN3K: A Dataset for Thyroid Nodule Segmentation in Ultrasound Images*. IEEE Transactions on Medical Imaging.
- **Redistribution Notice:** Contains non-commercial and share-alike terms. Included for research/demo purposes; users must verify the original dataset/model terms before redistribution.

---

## 3. Compliance and Attribution Declarations

1. **Permissive Open Source Alignment:** All integrated deep learning frameworks (`MONAI`, `nnUNet`, `TotalSegmentator`, `MedSAM`, `TorchXRayVision`) are licensed under Apache 2.0 or MIT, permitting integration, modification, and reproduction provided copyright notices and disclaimers are preserved.
2. **Demo Sample Notice:** All demo samples in `MURL_DEMO_DATA/` are included for research/demo purposes; users must verify the original dataset/model terms before redistribution.
3. **Non-Destructive Boundary:** Upstream libraries remain unedited and intact. All platform extensions are implemented through external thin adapters.
4. **Regulatory Research Use Notice:** All outputs generated by the MURL platform carry the mandatory academic disclaimer:
   > *"This platform is intended for research and educational purposes. Outputs are experimental and are not a medical diagnosis or a substitute for qualified clinical interpretation."*
