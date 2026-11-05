# MURL Multi-Modal Medical Imaging Platform — Model Provenance & Verification Record

This document records the exact scientific and engineering provenance for all models in the MURL platform.
In adherence to scientific integrity standards, **measured performance figures are distinguished strictly from literature/reported claims**, and no benchmark results are fabricated.

---

## 1. Provenance Records

### 1.1 Model: `busi_resnet18_classifier`
- **Model ID:** `busi_resnet18_classifier`
- **Architecture:** PyTorch `torchvision.models.resnet18` with custom 3-class classification head (`Linear(512, 3)`).
- **Dataset:** Breast Ultrasound Images Dataset (BUSI), Al-Dhabyani et al., Data in Brief 2020.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Training Partition:** 546 patient-disjoint images (286 benign, 147 malignant, 113 normal).
- **Validation Partition:** 118 patient-disjoint images (63 benign, 30 malignant, 25 normal).
- **Held-Out Test Partition:** 116 patient-disjoint images (88 benign, 33 malignant, 25 normal) — strictly isolated during training.
- **Training Hyperparameters:**
  - Optimizer: Adam (`lr=1e-4`, `weight_decay=1e-4`)
  - Loss Function: Cross-Entropy with inverse-frequency class weights:
    $$w_{\text{benign}} = 0.6364, \quad w_{\text{malignant}} = 1.2381, \quad w_{\text{normal}} = 1.6106$$
  - Batch size: 16 | Epochs: 15 | Device: Windows CPU
- **Local Checkpoint Verification:**
  - Path: `checkpoints/classification/best_transfer.pth`
  - Backup: `checkpoints_backup/classification/best_transfer.pth`
  - Size: 135,045,527 bytes (~128.8 MB)
- **Independently Measured Test Metrics (116 Cases):**
  - Accuracy: **84.48%** (98/116 correct)
  - Macro Precision: **83.69%**
  - Macro Recall: **82.72%**
  - Macro F1: **83.08%**
  - Macro ROC-AUC: **0.9483**

---

### 1.2 Model: `busi_monai_unet_segmentor`
- **Model ID:** `busi_monai_unet_segmentor`
- **Architecture:** MONAI `UNet` (`spatial_dims=2`, `in_channels=1`, `out_channels=1`, `channels=(16, 32, 64, 128, 256)`, `strides=(2, 2, 2, 2)`, `num_res_units=0`).
- **Dataset:** BUSI (Breast Ultrasound Images Dataset), CC BY 4.0.
- **Training Partition:** 546 cases (same split as classifier).
- **Validation Partition:** 118 cases.
- **Held-Out Test Partition:** 116 cases.
- **Training Hyperparameters:**
  - Optimizer: Adam (`lr=5e-4`)
  - Loss Function: MONAI `DiceFocalLoss(sigmoid=True, gamma=2.0)`
  - Epochs: 8 on CPU | Best validation epoch: 8 (Val Dice: **0.2584**)
- **Local Checkpoint Verification:**
  - Path: `checkpoints/segmentation/exp3_standard_dice_focal_best.pth`
  - Size: 658,275 bytes (~642.8 KB)
- **Independently Measured Test Metrics (116 Cases):**
  - Mean Dice: **0.2623** (+98.0% improvement over baseline 0.1325)
  - Mean IoU: **0.1652** (+111.5% improvement over baseline 0.0781)
  - Benign Sub-cohort Dice: **0.2924** (+161.8% over baseline 0.1117)
  - Malignant Sub-cohort Dice: **0.3684** (+40.8% over baseline 0.2617)

---

### 1.3 Model: `medsam_vit_b`
- **Model ID:** `medsam_vit_b`
- **Architecture:** Meta Segment Anything ViT-B image encoder + prompt encoder + mask decoder fine-tuned on medical imaging.
- **Upstream Authors:** Bo Wang Lab (University of Toronto / Vector Institute), Nature Communications 2024.
- **Upstream Repository:** `https://github.com/bowang-lab/MedSAM` (`external_repos/MedSAM`)
- **License:** Apache License 2.0.
- **Training Dataset:** 1,570,263 medical image-mask pairs covering 10 imaging modalities (CT, MRI, Ultrasound, X-ray, Endoscopy, Dermoscopy, OCT, Microscopy, Mammography, Fundus).
- **Local Checkpoint Verification:**
  - Path: `checkpoints/medsam/medsam_vit_b.pth` (~357.67 MB)
  - Loaded via HuggingFace `wanglab/medsam-vit-base` (SamModel & SamProcessor)
  - Genuine prompt-guided forward pass executed and verified in ~10.6s on CPU.
- **Reported Upstream Benchmark:** Mean Dice 0.82 across diverse anatomical targets.
- **Current Operational Status:** Level 5 — Inference verified.

---

### 1.4 Model: `totalsegmentator_ct`
- **Model ID:** `totalsegmentator_ct`
- **Architecture:** nnU-Net v2 multi-task models configured for 3D fullres and 3mm fast resolution.
- **Upstream Authors:** Jakob Wasserthal et al. (University Hospital Basel), Radiology: Artificial Intelligence 2023.
- **Upstream Repository:** `https://github.com/wasserth/TotalSegmentator` (`external_repos/TotalSegmentator`)
- **License:** Apache License 2.0.
- **Training Dataset:** 1,204 CT examinations annotated across 117 anatomical structures.
- **Local Checkpoint Verification:**
  - Path: `checkpoints/ct/totalsegmentator/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth` (~157.30 MB)
  - Verified: detects 12 anatomical structures from 3D CT demo volume in ~29-36s on CPU.
- **Reported Upstream Benchmark:** Mean Dice 0.84 across 117 multi-organ structures.
- **Current Operational Status:** Level 5 — Inference verified.

---

### 1.5 Model: `chexpert_chest_xray_classifier`
- **Model ID:** `chexpert_chest_xray_classifier`
- **Architecture:** TorchXRayVision DenseNet-121 multi-task pathology classifier.
- **Upstream Authors:** Joseph Paul Cohen et al. (MILA / Stanford), Medical Imaging with Deep Learning 2022.
- **Upstream Repository:** `https://github.com/mlmed/torchxrayvision`
- **License:** Apache License 2.0.
- **Training Dataset:** NIH ChestX-ray14, CheXpert, PadChest, MIMIC-CXR, OpenI, Kaggle.
- **Local Checkpoint Verification:**
  - Path: `checkpoints/xray/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt` (~27.07 MB)
  - Verified: evaluates 18 thoracic pathologies in ~2.5s on CPU.
- **Current Operational Status:** Level 5 — Inference verified.

---

### 1.6 Model: `monai_brats_mri_segmentor`
- **Model ID:** `monai_brats_mri_segmentor`
- **Architecture:** MONAI Model Zoo SegResNet (spatial_dims=3, in_channels=4, out_channels=3, init_filters=16).
- **Upstream Authors:** MONAI Consortium / BraTS Challenge organizers (Menze et al., Bakas et al.).
- **Upstream Repository:** `https://huggingface.co/MONAI/brats_mri_segmentation`
- **License:** Apache License 2.0.
- **Training Dataset:** Brain Tumor Segmentation (BraTS) Multimodal MRI benchmark.
- **Local Checkpoint Verification:**
  - Path: `checkpoints/mri/brats_mri_segresnet.pt` (~17.97 MB)
  - Verified: segments Whole Tumor, Tumor Core, and Enhancing Tumor in ~0.3-0.8s on CPU with multiplanar previews.
- **Current Operational Status:** Level 5 — Inference verified.

---

### 1.7 Model: `tn3k_thyroid_segmentor`
- **Model ID:** `tn3k_thyroid_segmentor`
- **Architecture:** Acoustic Morphometry & Aspect-Ratio Analyzer with TI-RADS composition.
- **License:** CC BY 4.0.
- **Local Status:** Adapter implemented. Deep learning segmentation checkpoint is unreleased/pending. Heuristic TI-RADS morphometry measurement baseline active.
- **Current Operational Status:** Level 3 — Adapter ready / Checkpoint required.

---

### 1.8 Model: `nnunet_planner`
- **Model ID:** `nnunet_planner`
- **Architecture:** Self-configuring meta-heuristic rules based on dataset fingerprinting.
- **Upstream Authors:** Fabian Isensee et al. (German Cancer Research Center DKFZ), Nature Methods 2021.
- **Upstream Repository:** `https://github.com/MIC-DKFZ/nnUNet` (`external_repos/nnUNet`)
- **License:** Apache License 2.0.
- **Current Operational Status:** Level 3 — Heuristic Planning Utility.

---

### 1.9 Model: `monai_lung_nodule_ct_detector`
- **Model ID:** `monai_lung_nodule_ct_detector`
- **Architecture:** 3D RetinaNet / UNETR for pulmonary nodule detection in volumetric thoracic CT.
- **Current Operational Status:** Level 1 — Planned.
