"""Reproducibility: seed all RNGs. Eval must be deterministic given seed."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch (CPU+CUDA). Set cudnn deterministic for eval.

    Args:
        seed: RNG seed.
        deterministic: if True, force cudnn deterministic + disable benchmark.
            Use True for eval; may be relaxed in training for speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
