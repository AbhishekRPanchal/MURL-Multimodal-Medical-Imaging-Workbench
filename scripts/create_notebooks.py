"""
Helper script to generate reproducible Jupyter notebooks
"""

import os
import json

def make_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")

os.makedirs("notebooks", exist_ok=True)

# 1. EDA Notebook
cells_1 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01. Breast Ultrasound Dataset (BUSI) Audit & Exploratory Data Analysis\n",
            "### MURL — Multimodal Medical Imaging Research Workbench\n",
            "\n",
            "This notebook demonstrates rigorous dataset provenance analysis, image-mask pairing verification, multi-mask fusion, and leak-free stratified split generation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "\n",
            "# Load audited metadata\n",
            "with open('data/processed/dataset_metadata.json') as f:\n",
            "    meta = json.load(f)\n",
            "\n",
            "print('Provenance:')\n",
            "for k, v in meta['provenance'].items():\n",
            "    print(f'  {k}: {v}')\n",
            "\n",
            "print('\\nAudit Summary:')\n",
            "for k, v in meta['audit_summary'].items():\n",
            "    print(f'  {k}: {v}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize class distributions across stratified splits\n",
            "classes = ['benign', 'malignant', 'normal']\n",
            "train_counts = [sum(1 for c in meta['cases'] if c['split'] == 'train' and c['class'] == cls) for cls in classes]\n",
            "val_counts = [sum(1 for c in meta['cases'] if c['split'] == 'val' and c['class'] == cls) for cls in classes]\n",
            "test_counts = [sum(1 for c in meta['cases'] if c['split'] == 'test' and c['class'] == cls) for cls in classes]\n",
            "\n",
            "x = np.arange(len(classes))\n",
            "width = 0.25\n",
            "\n",
            "plt.figure(figsize=(8, 5))\n",
            "plt.bar(x - width, train_counts, width, label='Train (70%)', color='#2563eb')\n",
            "plt.bar(x, val_counts, width, label='Val (15%)', color='#f59e0b')\n",
            "plt.bar(x + width, test_counts, width, label='Held-out Test (15%)', color='#10b981')\n",
            "\n",
            "plt.xlabel('Diagnostic Category')\n",
            "plt.ylabel('Number of Cases')\n",
            "plt.title('Case-Level Stratified Partitioning (BUSI)')\n",
            "plt.xticks(x, [c.capitalize() for c in classes])\n",
            "plt.legend()\n",
            "plt.grid(axis='y', alpha=0.3)\n",
            "plt.show()\n"
        ]
    }
]
make_notebook(cells_1, "notebooks/01_dataset_eda_and_audit.ipynb")

# 2. Classification Notebook
cells_2 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02. Ultrasound Classification Experiments & Benchmark Comparison\n",
            "### MURL — Multimodal Medical Imaging Research Workbench\n",
            "\n",
            "Empirical comparison between custom UltrasoundBaselineCNN and Pretrained ResNet18 Transfer Learning with Class-Weighted Objectives."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Load test evaluation results\n",
            "with open('outputs/test_evaluation_report.json') as f:\n",
            "    report = json.load(f)\n",
            "\n",
            "cls_rep = report['classification']\n",
            "for model_name, metrics in cls_rep.items():\n",
            "    print(f'=== {model_name.upper()} ===')\n",
            "    print(f'  Test Accuracy:    {metrics[\"accuracy\"]*100:.2f}%')\n",
            "    print(f'  Macro Precision:  {metrics[\"precision_macro\"]:.4f}')\n",
            "    print(f'  Macro Recall:     {metrics[\"recall_macro\"]:.4f}')\n",
            "    print(f'  Macro F1-Score:   {metrics[\"f1_macro\"]:.4f}')\n",
            "    print(f'  ROC-AUC (Macro):  {metrics[\"roc_auc_macro\"]:.4f}')\n"
        ]
    }
]
make_notebook(cells_2, "notebooks/02_classification_experiments.ipynb")

# 3. Segmentation Notebook
cells_3 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03. MONAI 2D Residual U-Net Lesion Segmentation\n",
            "### MURL — Multimodal Medical Imaging Research Workbench\n",
            "\n",
            "Evaluation of Project MONAI U-Net trained with DiceCELoss for pixel-level boundary segmentation on held-out test data."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "\n",
            "fig_path = 'outputs/figures/segmentation_test_comparisons.png'\n",
            "if os.path.exists(fig_path):\n",
            "    img = Image.open(fig_path)\n",
            "    plt.figure(figsize=(16, 12))\n",
            "    plt.imshow(img)\n",
            "    plt.axis('off')\n",
            "    plt.title('Test Set Segmentation Overlays (MONAI UNet)')\n",
            "    plt.show()\n"
        ]
    }
]
make_notebook(cells_3, "notebooks/03_segmentation_experiments.ipynb")
