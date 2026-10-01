"""Reproducibility helpers shared by the Telco churn experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and hash randomization for repeatable runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
