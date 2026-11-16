"""`train_phyclip_vit_b.py` on GRIT for 200K iterations; nothing else changes."""

from .grit_dataset import dataset  # noqa: F401
from .train_phyclip_vit_b import model, optim, train  # noqa: F401

train["num_iterations"] = 200000
