"""Random seeds and categorical feature encoding shared by CCMG modules."""

import random

import numpy as np
import torch


def seed_everything(seed: int):
    """Seed Python, NumPy, and PyTorch (including all visible CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def onehot_encoding_unk(value, allowable_set):
    """Encode a category, mapping unknown values to the final category."""
    if value not in allowable_set:
        value = allowable_set[-1]
    return [int(value == category) for category in allowable_set]
