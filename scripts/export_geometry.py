"""
Export the data behind the geometry figure and the operating points.

For each model, on the same fixed set of CC3M training samples:

  - the learned curvature against training iteration, read from the run's
    tensorboard scalars;
  - the distance from the origin of every image, text, box image and box text;
  - the operating point u = sinh(sqrt(c) * mean distance from the origin);
  - for text-image pairs, the text radius r_1 (mode of the text distribution)
    and the 1st to 99th percentile window of the image radius r_2 and of the
    angle at the origin, which is where the distance distortion is evaluated.

    python scripts/export_geometry.py --arch b --out results/geometry_vit_b.json

Each run is read from `train_results/<run>/`, named after its config.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import gaussian_kde
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gromovclip import euclidean as E
from gromovclip import lorentz as L
from gromovclip.config import LazyConfig, instantiate
from gromovclip.tokenizer import Tokenizer
from gromovclip.utils.checkpointing import CheckpointManager
from gromovclip.utils.directory import generate_output_dir_name

METHODS = ["hycoclip", "argent", "phyclip", "gromovclip"]
SIGNALS = ["image", "text", "box_image", "box_text"]


def read_curvature(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(iterations,)` and `(factors, iterations)` from a run's tensorboard scalars.

    A resumed run leaves one event file per launch, and the overlapping steps
    are rewrites of the same iteration, so the later file wins.
    """
    series: dict[str, dict[int, float]] = {}
    for events in sorted(run.glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if re.fullmatch(r"curv/curv(_\d+)?", tag):
                series.setdefault(tag, {}).update(
                    {point.step: point.value for point in acc.Scalars(tag)}
                )
    if not series:
        return np.empty(0), np.empty((0, 0))
    steps = np.array(sorted(set.intersection(*(set(v) for v in series.values()))))
    return steps, np.array([[series[tag][s] for s in steps] for tag in sorted(series)])


def factors(model, feats: torch.Tensor) -> torch.Tensor:
    """`(B, D)` -> `(B, num_factors, factor_dim)`, one factor for a single hyperboloid."""
    if hasattr(model, "num_subspaces"):
        return feats.view(-1, model.num_subspaces, model.subspace_dim)
    return feats.unsqueeze(1)


def combine(model, per_factor: torch.Tensor) -> torch.Tensor:
    """Collapse a per-factor reading over the last axis, as the product metric does."""
    if getattr(model, "product_metric", "l1") == "l1":
        return per_factor.mean(dim=-1)
    return per_factor.pow(2).mean(dim=-1).sqrt()


def curvature(model) -> float:
    """The learned curvature, averaged over the factors of a product model."""
    if hasattr(model, "curvs"):
        return float(torch.stack([curv.exp() for curv in model.curvs]).mean())
    return float(model.curv.exp())


def origin_distance(model, feats: torch.Tensor) -> torch.Tensor:
    """Distance from the origin per sample, as in the model's own logging."""
    subspaces = factors(model, feats)
    if hasattr(model, "curvs"):
        curvs = torch.stack([curv.detach().exp() for curv in model.curvs])
        per_factor = L.get_distance_from_origin(subspaces, curvs).squeeze(-1)
    elif hasattr(model, "curv"):
        per_factor = L.get_distance_from_origin(
            subspaces, model.curv.detach().exp()
        ).squeeze(-1)
    else:
        per_factor = E.get_distance_from_origin(subspaces).squeeze(-1)
    return combine(model, per_factor)


def origin_angles(model, parent, child, budget: int = 1 << 22) -> np.ndarray:
    """Angle at the origin between every parent and every child."""
    p, c = factors(model, parent), factors(model, child)
    p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    step = max(1, budget // (c.shape[0] * c.shape[1]))
    out = [
        combine(
            model,
            torch.einsum("pnd,cnd->pcn", p[i : i + step], c).clamp(-1, 1).arccos(),
        )
        for i in range(0, p.shape[0], step)
    ]
    return torch.cat([o.flatten() for o in out]).numpy()


def mode(x: np.ndarray, resolution: int = 512) -> float:
    """Peak of the density. The mean of a skewed radius distribution can fall
    where the model has no mass."""
    grid = np.linspace(x.min(), x.max(), resolution)
    return float(grid[np.argmax(gaussian_kde(x)(grid))])


@torch.inference_mode()
def embed(model, batches, device) -> dict[str, torch.Tensor]:
    tokenizer = Tokenizer()
    out = {s: [] for s in SIGNALS}
    for batch in batches:
        for signal in SIGNALS:
            if signal.endswith("image"):
                feats = model.encode_image(batch[signal].to(device), project=True)
            else:
                feats = model.encode_text(tokenizer(batch[signal]), project=True)
            out[signal].append(feats.float().cpu())
    return {s: torch.cat(v) for s, v in out.items()}


def fixed_batches(num_samples: int, batch_size: int) -> list[dict]:
    """The same training samples for every model."""
    cfg = LazyConfig.load("configs/cc3m_dataset.py")
    cfg.dataset.infinite_stream = False
    cfg.dataset.seed = 0
    loader = torch.utils.data.DataLoader(
        instantiate(cfg.dataset), batch_size=batch_size, num_workers=4
    )
    batches, taken = [], 0
    for batch in loader:
        batches.append(batch)
        taken += batch["image"].shape[0]
        if taken >= num_samples:
            break
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="b", choices=["s", "b"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)

    batches = fixed_batches(args.num_samples, args.batch_size)
    out = {"arch": args.arch, "num_samples": args.num_samples, "models": {}}

    for method in args.methods:
        cfg = LazyConfig.load(f"configs/train_{method}_vit_{args.arch}.py")
        run = Path(generate_output_dir_name(cfg))
        model = instantiate(cfg.model).to(device).eval()
        CheckpointManager(model=model).load(str(run / "checkpoint_final.pth"))

        steps, curvs = read_curvature(run)
        feats = embed(model, batches, device)
        model = model.cpu()
        radii = {s: origin_distance(model, feats[s]).numpy() for s in SIGNALS}
        theta = origin_angles(model, feats["text"], feats["image"])
        r2_lo, r2_hi = np.percentile(radii["image"], (1.0, 99.0))
        th_lo, th_hi = np.percentile(theta, (1.0, 99.0))
        c = curvature(model)
        u = {s: float(np.sinh(np.sqrt(c) * radii[s].mean())) for s in ("image", "text")}

        out["models"][method] = {
            "curvature": {
                "steps": steps.tolist(),
                "curvs": np.round(curvs, 5).tolist(),
                "floor": cfg.model.get("curv_init", 1.0) / 10,
            },
            "c_final": c,
            "operating_point": u,
            "radii": {s: np.round(d, 5).tolist() for s, d in radii.items()},
            "distortion": {
                "r1": mode(radii["text"]),
                "c": c,
                "y_lo": float(r2_lo),
                "y_hi": float(r2_hi),
                "th_lo": float(th_lo),
                "th_hi": float(th_hi),
            },
        }
        print(
            f"{method:<12} c={c:.3f}  u_image={u['image']:.2f}  u_text={u['text']:.2f}  "
            + "  ".join(f"{s}={d.mean():.3f}" for s, d in radii.items()),
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
