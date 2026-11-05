"""
Reproducibility Utilities
Ensures deterministic behavior across Python, NumPy, PyTorch, and MONAI.
"""

import os
import random
import numpy as np
import torch
import monai

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    monai.utils.set_determinism(seed=seed)
    print(f"[Reproducibility] Random seed fixed to {seed} across python, numpy, torch, monai.")

def get_device() -> torch.device:
    """Detect available hardware accelerator or fallback to CPU with logging."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using GPU: {device_name}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available. Running on CPU (deterministic CPU pipeline active).")
    return device
