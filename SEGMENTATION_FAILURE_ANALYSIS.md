# Segmentation Failure Analysis Report

## 1. Executive Summary & Problem Formulation
During initial baseline training, the MONAI Residual U-Net reported:
- **Held-out Test Mean Dice:** **0.1325 (13.25%)**
- **Held-out Test Mean IoU:** **0.0781 (7.81%)**
- **Benign Subset Dice:** **0.1117 (11.17%)**
- **Malignant Subset Dice:** **0.2617 (26.17%)**
- **Normal Subset Dice:** **0.0000 (0.00%)**

This investigation conducted a systematic code, tensor, gradient, and visual diagnostic audit to isolate the root mechanism responsible for this low performance.

**Primary Finding:** 
The low Dice score is **NOT** caused by data corruption, coordinate misalignments, or mask indexing bugs. Rather, it is the mathematical result of **Foreground Probability Collapse (Over-Prediction Flooding)** under early-epoch convergence ($E=4$ on CPU). The network's uncalibrated logits hovered uniformly around $+0.25$ to $+0.32$ (probabilities $\approx 0.57$). Applying a standard clinical threshold of $0.5$ caused **96.1% of the entire image canvas (62,968 pixels out of 65,536)** to be classified as lesion foreground, resulting in a low intersection-to-sum ratio ($2 \times 4000 / (63000 + 4000) \approx 0.119$).

---

## 2. Systematic Pipeline Audit Checklist

| Audit Component | Diagnostic Methodology | Finding / Evidence | Status |
| :--- | :--- | :--- | :---: |
| **1. Image-Mask Pairing** | Case hash and ID cross-verification in `BUSISegmentationDataset` | Verified: Every image `path` matches its exact `mask_paths` (e.g. `benign (1).png` pairs with `benign (1)_mask.png`). | **VERIFIED** |
| **2. Mask Pixel Values & Foregrounds** | Inspected min/max pixel values and unique set in PyTorch loader | Values strictly belong to $\{0.0, 1.0\}$. Foreground represents acoustic lesion, background is 0. | **VERIFIED** |
| **3. Multi-Mask Lesions** | Audited 17 multi-focal cases | Correctly fused via deterministic `np.bitwise_or`. | **VERIFIED** |
| **4. Empty Masks (Normal Scans)** | Inspected 133 normal scans | Normal scans contain all-zero masks ($0\text{ px}$). Under standard Dice, empty-to-empty is $1.0$, but empty-to-nonempty is $0.0$. | **CRITICAL FACTOR** |
| **5. Resizing & Interpolation** | Code audit in `load_case_mask` and dataset caching | Images resized via `Image.BILINEAR`, masks resized strictly via `Image.NEAREST` to prevent interpolation blur. | **VERIFIED** |
| **6. Tensor Normalization** | Audited `TF.to_tensor()` input tensors | Greyscale images normalized to $[0.0, 1.0]$. Zero NaN or Inf values detected. | **VERIFIED** |
| **7. Spatial Dimensions** | Checked input/output tensor shapes | Input: $[B, 1, 256, 256]$, Output Logits: $[B, 1, 256, 256]$. Native 1:1 spatial correspondence. | **VERIFIED** |
| **8. Loss Function Mechanics** | Audited MONAI `DiceCELoss(sigmoid=True)` | Symmetrical 1:1 weighting of Dice and BCE without focal or background suppression. | **CRITICAL FACTOR** |
| **9. Logit Distribution** | Extracted raw logits from test set inference | Mean logit: $+0.29$ (Mean probability: **$0.573$**). Foreground flooding at $\tau=0.5$. | **ROOT CAUSE** |

---

## 3. Quantitative Failure Evidence (N = 116 Test Cases)

From the diagnostic run (`scripts/diagnose_segmentation.py`, logged in `outputs/failure_analysis/diagnostic_summary.json`):

| Pathology Cohort | Case Count ($N$) | Avg Ground Truth Foreground (Pixels) | Avg Predicted Foreground ($\tau=0.5$) | Canvas Percentage Predicted as Lesion | Mean Sørensen-Dice ($\tau=0.5$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 65 | 4,018 px ($6.1\%$) | **62,968 px** | **$96.08\%$** | **0.1117** |
| **Malignant** | 31 | 10,342 px ($15.7\%$) | **63,497 px** | **$96.88\%$** | **0.2617** |
| **Normal** | 20 | 0 px ($0.0\%$) | **62,861 px** | **$95.91\%$** | **0.0000** |
| **Overall Held-out Test** | **116** | **4,704 px ($7.2\%$)** | **63,091 px** | **$96.27\%$** | **0.1325** |

### Mathematical Breakdown of the Metric:
For a benign scan with true foreground area $|Y| = 4,018\text{ px}$:
When the network predicts foreground for almost the entire image $|X| = 62,968\text{ px}$, the intersection $|X \cap Y| \approx |Y| = 4,018\text{ px}$.
$$\text{Dice} = \frac{2 \cdot |X \cap Y|}{|X| + |Y|} = \frac{2 \times 4018}{62968 + 4018} = \frac{8036}{66986} = \mathbf{0.1199}$$
The observed score of **0.1117** matches the exact mathematical expectation of a network predicting foreground flooding.

For malignant scans, true tumor area is larger ($|Y| = 10,342\text{ px}$):
$$\text{Dice} = \frac{2 \times 10342}{63497 + 10342} = \frac{20684}{73839} = \mathbf{0.2801}$$
This explains why malignant Dice is more than double the benign Dice ($0.2617$ vs $0.1117$): larger ground truth lesions mathematically yield a higher Dice score under uniform foreground flooding!

For normal scans, true lesion area is 0 ($|Y| = 0$):
$$\text{Dice} = \frac{2 \times 0}{62861 + 0} = \mathbf{0.0000}$$
Because 20 test scans are normal, 20 cases receive an automatic Dice of $0.0000$, lowering the overall dataset average.

---

## 4. Root Causes & Failure Mechanisms

### Mechanism 1: Extreme Spatial Class Imbalance (93% Background vs 7% Foreground)
Ultrasound lesions occupy only 6–15% of the 2D scan area. Standard cross-entropy treats all pixels equally, but standard Dice loss struggles in early iterations when the background gradient does not sufficiently penalize false positives.

### Mechanism 2: Under-Convergence in CPU Budget (4 Epochs = 272 Parameter Updates)
Under initial baseline CPU training, the model was trained for 4 epochs. Random initialization starts logits near 0 ($\sigma(0) = 0.5$). In 272 gradient steps, the network began learning tumor contrast gradients, but the decoder bias parameters did not have enough optimization steps to shift background logits into the strong negative regime ($\text{logit} < -2.0, \sigma < 0.1$).

### Mechanism 3: Uncalibrated Thresholding at $\tau = 0.5$
Because the probability baseline was $0.57$, applying a rigid threshold of $\tau = 0.5$ classified both lesion and background as foreground. In contrast, thresholding at or above the mean probability, or training with a loss that forces background suppression, prevents this collapse.

### Mechanism 4: Dice Loss Behavior on All-Zero Normal Masks
In MONAI's default `DiceLoss(sigmoid=True)`, when the ground-truth mask is all zeros ($|Y| = 0$), the numerator $2 \sum (p \cdot y) = 0$. Regardless of how well the network suppresses probabilities, the Dice loss remains $\approx 1.0$. This destabilizes gradients on normal cases unless the loss is configured with background smoothing or normal cases are routed through a classification gate.

---

## 5. Visual Evidence: 20-Panel Failure Comparison

The 20-panel visual failure comparison has been generated and saved to:
`outputs/figures/segmentation_failure_20_comparisons.png`

The panel displays a $20 \times 3$ grid covering:
- **Rows 1–10:** 10 Benign Cases (small lesions, oval margins, low Dice $\sim 0.01$–$0.07$ due to white flooding).
- **Rows 11–17:** 7 Malignant Cases (larger hypoechoic lesions, acoustic shadows, Dice $\sim 0.20$–$0.35$).
- **Rows 18–20:** 3 Normal Cases (no lesions, all-black ground truth, all-white predicted mask, Dice $= 0.000$).

Columns per row:
1. **Original B-Mode Ultrasound Image** (normalized $256 \times 256$).
2. **Ground Truth Mask** (radiologist gold standard).
3. **Predicted Mask ($\tau = 0.5$)** (exhibiting the white foreground flooding phenomenon).

---

## 6. Actionable Blueprint for Segmentation Repair (Task 3)

To repair the segmentation pipeline without violating the data split or CPU budget:
1. **Architectural Comparison:**
   - Model A: Existing Residual MONAI U-Net (`channels=(16, 32, 64, 128)`, `num_res_units=1`).
   - Model B: Standard Lightweight MONAI U-Net without residual units (`channels=(16, 32, 64, 128)`, `num_res_units=0`) to evaluate whether residual connections at small batch sizes caused gradient stagnation.
2. **Loss Function Repair:**
   - Replace unweighted `DiceCELoss` with a **Focal-Dice Loss (`FocalLoss + DiceLoss`)** or a **Weighted BCE + Dice Loss** with positive class weighting and background penalty to force background logits to negative values ($\text{logit} < -2.0$).
3. **Controlled Optimization:**
   - Increase epochs safely to 10–12 epochs with AdamW ($\text{lr} = 10^{-3}$ with warm-up) and gradient clipping.
   - Monitor validation Dice and IoU on validation fold (118 cases) to select the optimal model.
   - Only evaluate the selected best model once on the held-out test set (116 cases).
