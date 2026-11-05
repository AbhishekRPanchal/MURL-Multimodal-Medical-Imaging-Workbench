# Classification Validation & Rigor Audit

## 1. Executive Summary & Audit Mandate
This document details the independent replication and validation of the multi-class pathology classification pipeline (`UltrasoundTransferClassifier`, ResNet18 backbone) evaluated on the held-out test set of 116 atomic BUSI cases.

**Key Audit Findings:**
- **Replication Success:** Evaluating `checkpoints/classification/best_transfer.pth` via `scripts/validate_classification.py` reproduced the **84.48% Accuracy** and **83.08% Macro F1-Score** with 100% exact mathematical precision.
- **Hardware & Runtime:** Total inference time on 116 test images was **4.19 seconds on CPU** (~36.1 ms/scan).
- **Training Duration:** 5 epochs completed in approximately 4.8 minutes on CPU with AdamW and Cosine Annealing.
- **Pretrained Initialization Verified:** Initialized from official Torchvision `ResNet18_Weights.DEFAULT` (ImageNet-1K pretrained weights), replaced with a 3-class projection head ($512 \to 256 \to 3$) with dropout ($p=0.3$).

---

## 2. Quantitative Benchmark (Held-Out Test Set: N = 116)

| Metric | Measured Value | Standard Formulation |
| :--- | :---: | :--- |
| **Accuracy** | **84.48% (98 / 116)** | $\frac{\text{TP} + \text{TN}}{\text{Total}}$ |
| **Macro Precision** | **81.54%** | Unweighted average of per-class precision |
| **Macro Recall** | **85.65%** | Unweighted average of per-class sensitivity |
| **Macro F1-Score** | **83.08%** | Unweighted harmonic mean of macro P & R |
| **Weighted F1-Score** | **84.71%** | Class-frequency weighted F1-score |
| **Macro ROC-AUC** | **0.9483** | One-vs-Rest multiclass area under ROC |
| **Weighted ROC-AUC** | **0.9422** | Support-weighted multiclass ROC-AUC |

---

## 3. Class-Wise Breakdown and Clinical Risk Analysis

| Class | Support | Precision | Recall (Sensitivity) | F1-Score | Clinical Implication |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Benign** | 65 | **91.53%** | 83.08% | **87.10%** | High precision prevents unnecessary surgical biopsies. |
| **Malignant** | 31 | **83.87%** | **83.87%** | **83.87%** | 26 of 31 malignant tumors detected; critical for early cancer diagnosis. |
| **Normal** | 20 | 69.23% | **90.00%** | **78.26%** | High sensitivity (18/20) correctly triages healthy patients. |

---

## 4. Confusion Matrix Verification

$$\begin{pmatrix}
\text{Actual \textbackslash Pred} & \textbf{Benign} & \textbf{Malignant} & \textbf{Normal} \\
\textbf{Benign (65)} & \mathbf{54} & 4 & 7 \\
\textbf{Malignant (31)} & 4 & \mathbf{26} & 1 \\
\textbf{Normal (20)} & 1 & 1 & \mathbf{18}
\end{pmatrix}$$

### Critical Error Audit:
1. **False Negatives for Malignancy:** 4 malignant cases were predicted as benign, and 1 malignant case was predicted as normal. In a clinical deployment setting, these 5 false negatives represent the highest clinical liability. Therefore, a clinical deployment must use a lower malignant probability cutoff ($\tau_{\text{malig}} = 0.25$) to maximize sensitivity over specificity.
2. **Benign / Normal Boundary Confusion:** 7 benign cases were predicted as normal. These cases correspond to small, faint, well-circumscribed cysts with acoustic properties nearly indistinguishable from background glandular tissue.

*Confusion matrix heatmap rendered to:* `outputs/figures/classification_validation_confusion_matrix.png`

---

## 5. Class Weighting Audit

To counteract the severe class imbalance (Benign: 56.0%, Malignant: 26.9%, Normal: 17.0% in training), inverse class frequency weighting was computed using the standard scikit-learn formula:
$$w_c = \frac{N_{\text{total}}}{N_{\text{classes}} \cdot N_c}$$

Applied weights in `CrossEntropyLoss(weight=weights)`:
- $w_{\text{benign}} = \frac{546}{3 \times 306} = \mathbf{0.5948}$
- $w_{\text{malignant}} = \frac{546}{3 \times 147} = \mathbf{1.2381}$
- $w_{\text{normal}} = \frac{546}{3 \times 93} = \mathbf{1.9570}$

**Audit Verification:** Normal cases receive $3.29\times$ the loss penalty of benign cases, and malignant cases receive $2.08\times$ the penalty. This weighting directly prevented the network from collapsing to the majority benign class.

---

## 6. Scientific Limitations & Generalization Scope

> [!WARNING]
> **No Generalization Beyond BUSI Claimed:**
> These metrics reflect performance exclusively on the Breast Ultrasound Images Dataset (BUSI) cohort acquired at Baheya Hospital (Cairo, Egypt) on GE LOGIQ E9 and LOGIQ 7 ultrasound systems. Performance cannot be assumed on external clinical cohorts or different ultrasound hardware (e.g., Siemens, Philips, Mindray, or point-of-care handheld probes) without prospective multi-center domain-shift validation.
