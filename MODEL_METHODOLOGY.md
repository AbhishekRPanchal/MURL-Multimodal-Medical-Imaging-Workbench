# Deep Learning Model Methodology & Mathematical Formulations

## 1. Classification Architecture

### 1.1 Custom Ultrasound Baseline CNN (`UltrasoundBaselineCNN`)
Ultrasound images are characterized by acoustic speckle noise, low signal-to-noise ratio, and indistinct lesion boundaries. To establish a native baseline, we designed a 4-stage convolutional neural network:

$$\text{Stage } k: \quad \mathbf{X}_{k} = \text{MaxPool}_{2\times 2} \Big( \text{LeakyReLU}_{0.1} \big( \text{BatchNorm} (\mathbf{W}_k * \mathbf{X}_{k-1}) \big) \Big)$$

- **Stage 1 (3 $\to$ 32 filters, $3\times 3$):** Captures high-frequency acoustic speckle textures.
- **Stage 2 (32 $\to$ 64 filters, $3\times 3$):** Extracts intermediate hypoechoic tissue contrast.
- **Stage 3 (64 $\to$ 128 filters, $3\times 3$):** Detects lesion boundary margins and acoustic shadows.
- **Stage 4 (128 $\to$ 256 filters, $3\times 3$):** High-level semantic representation.
- **Feature Aggregation & Classifier:** $\text{AdaptiveAvgPool2D}(1 \times 1) \to \text{Dropout}(0.3) \to \text{Linear}(256, 128) \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(128, 3)$.

### 1.2 Transfer Learning Architecture (`UltrasoundTransferClassifier`)
To leverage rich hierarchical visual representations, we fine-tune a deep convolutional residual network (**ResNet18**) initialized with ImageNet parameters.
- **Residual Formulation:** $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$, which mitigates degradation in deep networks.
- **Head Adaptation:** The final 1000-way linear layer is replaced with an ultrasound-specific projection block:
  $$\hat{\mathbf{z}} = \mathbf{W}_2 \cdot \text{ReLU}\big(\text{Dropout}_{0.3}(\mathbf{W}_1 \mathbf{f}_{\text{backbone}} + \mathbf{b}_1)\big) + \mathbf{b}_2, \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 512}, \; \mathbf{W}_2 \in \mathbb{R}^{3 \times 128}$$

### 1.3 Class-Weighted Loss Formulation
To correct for the substantial class imbalance (Benign: 56.0%, Malignant: 26.9%, Normal: 17.0%), we compute inverse-frequency weights:

$$w_c = \frac{N}{C \cdot N_c}$$

The weighted cross-entropy objective is defined as:

$$\mathcal{L}_{\text{CE}}(\boldsymbol{\theta}) = -\frac{1}{B} \sum_{i=1}^B w_{y_i} \log\left( \frac{\exp(z_{i, y_i})}{\sum_{j=1}^C \exp(z_{i, j})} \right)$$

---

## 2. Lesion Segmentation Architecture (Project MONAI)

### 2.1 MONAI 2D Residual U-Net (`UltrasoundUNet`)
Medical image segmentation requires precise boundary localization. We adopt **Project MONAI's** 2D U-Net with residual sub-units (`monai.networks.nets.UNet`).

- **Encoder Path:** 4 downsampling blocks with feature channel capacities of $[16, 32, 64, 128, 256]$ and stride 2 convolutions.
- **Residual Units (`num_res_units=2`):** Each block incorporates internal residual skip connections to facilitate gradient backpropagation across deep feature maps.
- **Decoder Path:** 4 upsampling transpose-convolution blocks paired with skip connections from matching encoder depths to restore spatial resolution.
- **Bottleneck:** Receptive field covers the entire lesion area ($256 \times 256$ input).

### 2.2 Combined Loss Objective: MONAI `DiceCELoss`
Ultrasound lesions exhibit irregular shapes and variable sizes. Relying solely on binary cross-entropy leads to boundary blurring, while pure Dice loss can suffer from optimization instability during early epochs. We utilize MONAI's combined `DiceCELoss`:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}}$$

Where the soft Sørensen-Dice loss is defined as:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^{H \times W} p_i g_i + \epsilon}{\sum_{i=1}^{H \times W} p_i^2 + \sum_{i=1}^{H \times W} g_i^2 + \epsilon}$$

and binary cross-entropy is:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{H \times W} \sum_{i=1}^{H \times W} \Big[ g_i \log(p_i) + (1 - g_i) \log(1 - p_i) \Big]$$

where $p_i = \sigma(z_i) \in [0, 1]$ is the predicted sigmoid probability at pixel $i$, and $g_i \in \{0, 1\}$ is the fused binary ground truth.

---

## 3. Optimization Protocol

| Parameter | Classification (Baseline & ResNet18) | Segmentation (MONAI U-Net) |
| :--- | :---: | :---: |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Initial Learning Rate** | $3 \times 10^{-4}$ | $5 \times 10^{-4}$ |
| **Weight Decay** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **LR Schedule** | Cosine Annealing ($\eta_{\min} = 10^{-6}$) | Cosine Annealing ($\eta_{\min} = 10^{-6}$) |
| **Batch Size** | 32 (Baseline) / 16 (Transfer) | 8 |
| **Input Dimensions** | $3 \times 256 \times 256$ (ImageNet normalized) | $1 \times 256 \times 256$ (Grayscale $[0, 1]$) |
| **Target Checkpointing** | Validation Macro F1 | Validation Mean Sørensen-Dice |
| **Reproducibility Seed** | Fixed to 42 | Fixed to 42 |
