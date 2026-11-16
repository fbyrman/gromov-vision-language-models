"""
Embed a list of texts with a trained model, for the qualitative analysis.

Stores the tangent vectors at the origin (text encoder output times the learned
scale, before the exponential map) and the learned curvature. For PHyCLIP the
vectors are kept per factor, with one curvature per factor.

    python scripts/embed_texts.py \
        --train-config configs/train_gromovclip_vit_b.py \
        --checkpoint-path train_results/<run>/checkpoint_final.pth \
        --texts assets/qualitative/text_trees.json assets/qualitative/background_texts.json \
        --out outputs/qualitative/gromovclip.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gromovclip.config import LazyConfig, instantiate
from gromovclip.models import PHyCLIP
from gromovclip.tokenizer import Tokenizer
from gromovclip.utils.checkpointing import CheckpointManager


def read_texts(path: str) -> list[str]:
    """A JSON list of texts, or the text trees file, whose nodes hold the texts."""
    data = json.load(open(path))
    if isinstance(data, list):
        return data
    return [node["text"] for tree in data["trees"] for node in tree["nodes"]]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    texts = sorted({t for path in args.texts for t in read_texts(path)})

    cfg = LazyConfig.load(args.train_config)
    model = instantiate(cfg.model).eval()
    CheckpointManager(model=model).load(args.checkpoint_path)

    tokenizer = Tokenizer()
    feats = torch.cat(
        [
            model.encode_text(tokenizer(texts[i : i + 256]), project=False)
            for i in range(0, len(texts), 256)
        ]
    ).float()

    if isinstance(model, PHyCLIP):
        feats = feats.view(len(texts), model.num_subspaces, model.subspace_dim)
        alphas = torch.stack([a.exp() for a in model.textual_concept_alphas])
        feats = feats * alphas.view(1, -1, 1)
        curv = torch.stack([c.exp() for c in model.curvs]).flatten().float()
    else:
        feats = feats * model.textual_alpha.exp()
        curv = float(model.curv.exp())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"texts": texts, "text": feats, "curvature": curv}, args.out)
    print(f"wrote {args.out}: {tuple(feats.shape)}")


if __name__ == "__main__":
    main()
