# Dataset Documentation & Provenance Audit: Breast Ultrasound Images (BUSI)

## 1. Overview & Provenance

The **Breast Ultrasound Images (BUSI)** dataset is a clinical medical ultrasound benchmark collected and curated by Dr. Walid Al-Dhabyani, Mohammed Gomaa, Hussien Khaled, and Aly Fahmy at Cairo University and the National Cancer Institute, Egypt.

- **Primary Publication:** Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). *Dataset of breast ultrasound images*. **Data in Brief**, 28, 104863.
- **Digital Object Identifier (DOI):** [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)
- **Original Repository Host:** Mendeley Data (`10.17632/wmy84gzngw.1`)
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Clinical Acquisition Devices:** GE Healthcare LOGIQ E9 Ultrasound System and LOGIQ 7 Ultrasound System (operating at 1 to 5 MHz transducers).
- **Patient Cohort:** 600 female patients presenting for breast examination, aged between 25 and 75 years, collected in 2018.

---

## 2. Directory Structure & Ground-Truth Alignment

The raw distribution contains three primary diagnostic directories:
```
data/raw/Dataset_BUSI_with_GT/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   ├── ...
│   ├── benign (100).png
│   ├── benign (100)_mask.png
│   └── benign (100)_mask_1.png     <-- Multi-lesion mask
├── malignant/
│   ├── malignant (1).png
│   ├── malignant (1)_mask.png
│   └── ...
└── normal/
    ├── normal (1).png
    ├── normal (1)_mask.png          <-- Blank mask (no lesion)
    └── ...
```

---

## 3. Automated Integrity Audit Findings

The automated dataset audit executed via `src/data_audit.py` produced the following verified statistics:

| Metric | Benign | Malignant | Normal | Total Dataset |
| :--- | :---: | :---: | :---: | :---: |
| **Original B-Mode Scans** | 437 | 210 | 133 | **780** |
| **Mask Files** | 454 | 211 | 133 | **798** |
| **Cases with Multi-Masks** | 16 | 1 | 0 | **17** |
| **Corrupted / Unreadable Files** | 0 | 0 | 0 | **0** |
| **Dimension Mismatches** | 0 | 0 | 0 | **0** |
| **Minimum Resolution** | 201 × 310 | 341 × 442 | 190 × 425 | **190 × 310** |
| **Maximum Resolution** | 903 × 911 | 845 × 889 | 1048 × 719 | **1048 × 911** |
| **Mean Resolution** | 562 × 504 | 582 × 491 | 703 × 512 | **615.7 × 501.5** |

### Critical Quality Findings:
1. **Multi-Focal Lesions:** 17 patient scans contain multiple ground truth mask files (e.g. `benign (100)_mask.png` and `benign (100)_mask_1.png`). Our pipeline implements a deterministic logical OR fusion operator (`load_case_mask()`) to aggregate all disjoint lesions into a unified ground-truth binary mask without loss of annotation data.
2. **Normal Class Masks:** All 133 normal cases contain all-black (all-zero) masks, reflecting the true physiological absence of focal breast lesions.
3. **SHA-256 Hash Duplicate Analysis:**
   - `normal (56)_mask.png` == `normal (62)_mask.png` and `normal (63)_mask.png` == `normal (74)_mask.png`: Verified to be identical blank (0-intensity) arrays.
   - `benign (433).png` is identical in bytes to `malignant (145).png`: An upstream dataset annotation artifact from the original distribution where an image was duplicated across categories. Documented here to preserve transparency.

---

## 4. Patient-Level Stratified Train / Validation / Test Split

To strictly eliminate data leakage, partitioning is performed at the atomic case level (`{class}_{case_id}`). All masks, augmented views, and derived features for a given scan remain strictly within their assigned partition.

- **Split Ratio:** 70% Train, 15% Validation, 15% Held-out Test
- **Random Seed:** Fixed to 42 for exact reproducibility

| Split | Benign | Malignant | Normal | Total Cases | Percentage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 306 | 147 | 93 | **546** | 70.0% |
| **Validation** | 66 | 32 | 20 | **118** | 15.1% |
| **Held-out Test** | 65 | 31 | 20 | **116** | 14.9% |
| **Total** | 437 | 210 | 133 | **780** | 100.0% |

---

## 5. Class Imbalance Handling

The class distribution in the training set exhibits substantial class imbalance:
- **Benign:** 306 samples (56.0%)
- **Malignant:** 147 samples (26.9%)
- **Normal:** 93 samples (17.0%)

### Imbalance Mitigation Strategies Implemented:
1. **Inverse Class Frequency Weights:** Loss weights $w_c = \frac{N_{\text{total}}}{C \cdot N_c}$ applied to `CrossEntropyLoss`:
   - Benign: $w_0 = 0.595$
   - Malignant: $w_1 = 1.238$
   - Normal: $w_2 = 1.957$
2. **Ultrasound Domain-Specific Augmentation:** Horizontal flip, mild rotation ($-10^\circ$ to $+10^\circ$), and acoustic gain / contrast variation applied on the fly to low-frequency minority classes during training.
3. **Macro-Averaged Evaluation:** Macro F1 and class-specific ROC-AUC are used as primary optimization criteria rather than raw overall accuracy, preventing the majority benign class from masking poor performance on malignant or normal scans.
