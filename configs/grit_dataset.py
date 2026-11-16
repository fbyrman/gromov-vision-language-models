"""Shared GRIT grounded training dataset, imported by every GRIT config.

Reads GRIT webdataset shards at `GRIT_DIR`. One box crop and its phrase are
sampled per image per step. Same image transform as the CC3M configs.

The shards and their loader, `gromovclip.data.grit_grounded`, will be shared
upon publication; until then this config, and every GRIT config that imports
it, cannot be loaded.
"""

import os
from pathlib import Path

from torchvision import transforms as T

from gromovclip.config import LazyCall as L
from gromovclip.data.grit_grounded import GRITGroundedTarMapper, GRITGroundedWebDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]

_mapper = L(GRITGroundedTarMapper)(
    image_transform=[
        L(T.RandomResizedCrop)(
            size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
        ),
        L(T.ToTensor)(),
    ],
)

_data_dir = os.environ.get("GRIT_DIR", str(_REPO_ROOT / "datasets" / "grit"))
dataset = L(GRITGroundedWebDataset)(
    tarfiles=[os.path.join(_data_dir, "*.tar")],
    mapper=_mapper,
    buffer_size=4000,
    seed="${..train.seed}",
)
