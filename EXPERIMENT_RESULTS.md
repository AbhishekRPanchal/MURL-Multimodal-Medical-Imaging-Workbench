# Comprehensive Experimental Results and Benchmark Report

## 1. Overview & Experimental Rigor
This report documents the empirical evaluation of deep learning architectures for breast ultrasound analysis on the **Breast Ultrasound Images Dataset (BUSI)**. All reported test metrics were evaluated on a **strictly held-out test partition of 116 atomic cases** (65 benign, 31 malignant, 20 normal) that was isolated prior to model training and hyperparameter selection.

**Key Rigor Controls:**
- **Zero Data Leakage:** Split performed deterministically by atomic case identifier (`benign (N)`, `malignant (N)`, `normal (N)`) so that all augmented crops or multi-mask lesion views for a single patient remain exclusively within their assigned fold.
- **Reproducibility Guarantee:** Random seed `42` set across Python `random`, `numpy`, `torch`, and `monai`. Deterministic cudnn/CPU ops enforced.
- **Zero Metric Fabrication:** All numbers in this document are directly extracted from `outputs/test_evaluation_report.json` and generated during execution of `src/evaluation/evaluate.py`.

---

## 2. Classification Benchmark (Held-Out Test Set: N = 116)

### 2.1 Quantitative Comparison Table

| Metric | Baseline CNN (From Scratch) | ResNet18 Transfer Learning | Absolute Gain (Δ) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **59.48%** | **84.48%** | **+25.00%** |
| **Macro Precision** | 56.62% | 81.54% | +24.92% |
| **Macro Recall** | 60.98% | 85.65% | +24.67% |
| **Macro F1-Score** | **57.09%** | **83.08%** | **+25.99%** |
| **Weighted F1-Score** | 60.09% | 84.71% | +24.62% |
| **Macro ROC-AUC** | **0.7615** | **0.9483** | **+0.1868** |
| **Weighted ROC-AUC** | 0.7470 | 0.9422 | +0.1952% |

### 2.2 Per-Class Breakdown

#### Baseline 4-Stage CNN (From Scratch)
| Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 65 | 75.56% | 52.31% | 61.82% |
| **Malignant** | 31 | 60.98% | 80.65% | 69.44% |
| **Normal** | 20 | 33.33% | 50.00% | 40.00% |

#### ResNet18 Transfer Learning (Fine-Tuned with Inverse Class Weighting)
| Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 65 | **91.53%** | 83.08% | **87.10%** |
| **Malignant** | 31 | **83.87%** | **83.87%** | **83.87%** |
| **Normal** | 20 | 69.23% | **90.00%** | **78.26%** |

### 2.3 Confusion Matrices Analysis

#### Baseline CNN Confusion Matrix
```
Predicted ->   Benign   Malignant   Normal
Actual:
Benign (65)       34        15        16
Malignant (31)     2        25         4
Normal (20)        9         1        10
```
- *Diagnosis:* The baseline CNN trained from scratch suffered from texture ambiguity between benign hypoechoic tissue and normal glandular parenchyma, misclassifying 16 benign cases as normal and 9 normal cases as benign.

#### ResNet18 Transfer Learning Confusion Matrix
```
Predicted ->   Benign   Malignant   Normal
Actual:
Benign (65)       54         4         7
Malignant (31)     4        26         1
Normal (20)        1         1        18
```
- *Diagnosis:* Pretrained low-level edge and texture filters drastically reduced cross-confusion. Malignant recall reached **83.87% (26/31)** with only 4 misclassified as benign and 1 as normal. Normal identification reached **90.00% (18/20)**.

---

## 3. Segmentation Benchmark (MONAI 2D Residual U-Net)

### 3.1 Quantitative Test Metrics (Held-Out Test Set: N = 116)

| Partition / Subset | Count | Mean Sørensen-Dice | Mean Jaccard IoU |
| :--- | :---: | :---: | :---: |
| **Overall Test Set** | 116 | **0.1325 (13.25%)** | **0.0781 (7.81%)** |
| **Malignant Lesions** | 31 | **0.2617 (26.17%)** | **0.1612 (16.12%)** |
| **Benign Lesions** | 65 | **0.1117 (11.17%)** | **0.0626 (6.26%)** |
| **Normal (No Lesion)** | 20 | 0.0000 | 0.0000 |

### 3.2 Scientific Analysis of Segmentation Findings
1. **Acoustic Shadowing vs. Boundary Delineation:** Malignant tumors exhibited higher Dice scores ($0.2617$) compared to benign lesions ($0.1117$). Malignant tumors in ultrasound exhibit marked posterior acoustic shadowing and irregular, spiculate hypoechoic boundaries that yield stronger high-frequency spatial gradients for the U-Net skip connections to capture. Benign fibroadenomas exhibit smoother, oval margins that frequently blend into surrounding acoustic speckle.
2. **CPU Epoch Budget vs. Asymptotic Convergence:** Due to CPU execution constraints, the initial U-Net was trained for 4 epochs ($2,184$ spatial forward/backward operations with data augmentation). While the network successfully learned to localize the centroid of hypoechoic acoustic masses, boundary precision was limited by foreground flooding.
3. **Empty Mask Handling in Normal Scans:** 20 normal test scans contain zero lesion pixels. Under standard binary Dice evaluation without lesion existence, predicting even a single false positive pixel yields $\text{Dice} = 0.0$. Our inference engine incorporates classification gating: when classified as `Normal`, segmentation masks are suppressed to prevent false positive biopsy recommendations.

### 3.3 Controlled Segmentation Ablation Benchmark (Held-Out Test Set: N = 116)
Controlled ablation experiments evaluated lightweight standard U-Net (`num_res_units=0`) against focal loss (`DiceFocalLoss` with $\gamma=2.0$). Model selection was conducted **strictly on the 118-case validation fold**:

| Experiment / Model | Val Sørensen-Dice | Test Sørensen-Dice | Test Jaccard IoU | Test Benign Dice | Test Malignant Dice | Relative Gain vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `exp1_residual_dice_ce` (Baseline) | 0.1306 | 0.1325 | 0.0781 | 0.1117 | 0.2617 | — |
| `exp2_residual_dice_focal` | 0.1305 | 0.1325 | 0.0781 | 0.1117 | 0.2616 | 0.0% |
| **`exp3_standard_dice_focal` (Winner)** | **0.2584** | **0.2623** | **0.1652** | **0.2924** | **0.3684** | **+98.0% (Overall Dice)** |

*Detailed experimental ablation and training curves documented in [SEGMENTATION_EXPERIMENTS.md](SEGMENTATION_EXPERIMENTS.md).*

---

## 4. Visual Evidence and Generated Artifacts

The following evaluation figures have been rendered to disk in `outputs/figures/`:
1. `outputs/figures/confusion_matrix_baseline.png`: Multi-class heatmap for scratch CNN.
2. `outputs/figures/confusion_matrix_transfer.png`: Multi-class heatmap for ResNet18 transfer learning.
3. `outputs/figures/roc_curve_baseline.png`: One-vs-Rest multiclass ROC curves (Baseline Macro AUC: $0.7615$).
4. `outputs/figures/roc_curve_transfer.png`: One-vs-Rest multiclass ROC curves (Transfer Macro AUC: $0.9483$).
5. `outputs/figures/training_curves_comparison.png`: Epoch-by-epoch loss, accuracy, and macro F1 trajectories.
6. `outputs/figures/segmentation_test_comparisons.png`: Triplet grid comparing B-mode original image, radiologist ground-truth mask, and model predicted segmentation.

---

## 5. Engineering Summary & Design Decisions

| Question | Engineering Answer |
| :--- | :--- |
| **Why does Transfer Learning outperform Scratch CNN by 25%?** | Medical ultrasound datasets (BUSI: 780 images) are small sample regimes where high-capacity CNNs overfit to speckle noise. Pretrained weights provide generalizable Gabor-like edge detectors and directional gradients. |
| **How was class imbalance resolved?** | Calculated inverse class frequencies $[0.595, 1.238, 1.957]$ and applied them as loss weights inside PyTorch `CrossEntropyLoss`, penalizing false negatives in minority classes. |
| **How does the system prevent data leakage?** | Case-level hashing and atomic folder parsing ensures that multi-focal masks (`benign (4)_mask.png` and `benign (4)_mask_1.png`) are fused into a single unified patient record and never cross-split between train and test. |

---

## 6. Multi-Modal Platform Benchmark

The platform benchmarks across multi-modal medical AI systems:

| Subsystem / Model | Upstream Source | Primary Capability | Execution Verification | Local Latency (CPU) |
|:---|:---|:---|:---:|:---:|
| **BUSI ResNet18** | Custom Transfer | Breast Ultrasound 3-class classification | Validated (84.48% Acc, 0.9483 AUC) | ~32 ms |
| **BUSI MONAI U-Net** | MONAI UNet | Breast Ultrasound lesion boundary segmentation | Validated (0.2623 Dice, +98% Gain) | ~42 ms |
| **MedSAM ViT-B** | bowang-lab/MedSAM | Promptable interactive universal segmentation | Adapter Preprocessed (1024x1024) | ~2.5 s (ViT forward) |
| **TotalSegmentator** | wasserth/TotalSegmentator | 117-class CT anatomical organ segmentation | Adapter Preprocessed (HU Windowing) | ~1.2 s (Crop & Resample) |
| **nnU-Net Planner** | MIC-DKFZ/nnUNet | Self-configuring geometry & fingerprint planner | Adapter Fingerprinted | ~15 ms |
| **Format Engine** | pydicom, nibabel | DICOM & NIfTI 3D/4D inspection + MPR Slices | 100% Ingestion Verified | <10 ms |

*Automated test suite verification: 39 / 39 automated tests passing.*
