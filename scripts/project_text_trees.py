"""
Project the text trees into the text plane of each model.

The text plane of a model is fitted by HoroPCA (two components, the origin kept
at the centre of the disk, best of three restarts) on the background texts, not
on the trees. The trees are then projected into it. A projection to a plane
does not preserve angles, so the exterior angle of every step from a parent to
a child is measured in the full embedding space and stored next to the
positions.

PHyCLIP embeds a text in a product of hyperboloids and has no single plane.
Each of its trees is drawn in the factor in which the tree lies farthest from
the origin, with the plane fitted on the background texts in that factor, and
the exterior angle is measured in that factor.

    python scripts/project_text_trees.py \
        --horopca-dir /path/to/HoroPCA \
        --embeddings gromovclip=outputs/qualitative/gromovclip.pt \
                     hycoclip=outputs/qualitative/hycoclip.pt \
                     argent=outputs/qualitative/argent.pt \
                     phyclip=outputs/qualitative/phyclip.pt \
        --out results/text_trees.json

The embeddings come from `scripts/embed_texts.py`. `--horopca-dir` is a clone
of github.com/HazyResearch/HoroPCA.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

TREES = "assets/qualitative/text_trees.json"
BACKGROUND = "assets/qualitative/background_texts.json"


def to_ball(v: torch.Tensor, c: float) -> torch.Tensor:
    """Tangent vector at the origin -> Poincare ball of unit curvature."""
    n = v.norm(dim=-1, keepdim=True)
    return torch.tanh(math.sqrt(c) * n / 2) * v / n


def ball_distance(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Pairwise distance in the Poincare ball, from the Gram matrix."""
    sq = x.pow(2).sum(-1)
    diff = (sq[:, None] + sq[None, :] - 2 * x @ x.T).clamp_min(0)
    den = (1 - sq).clamp_min(1e-15)
    arg = 1 + 2 * diff / (den[:, None] * den[None, :])
    d = torch.acosh(arg.clamp_min(1 + 1e-12)) * (1 - torch.eye(len(x), dtype=x.dtype))
    return d.unsqueeze(-1) if keepdim else d


def load_horopca(repo: str):
    sys.path.insert(0, repo)
    # The reference code predates torch.linalg.
    torch.solve = lambda B, A: (torch.linalg.solve(A, B), None)
    import geom.poincare as poincare

    poincare.pairwise_distance = ball_distance
    from learning.pca import HoroPCA

    return HoroPCA


def fit(HoroPCA, z: torch.Tensor, seed: int = 0, restarts: int = 3, steps: int = 400):
    """HoroPCA to two components, without Frechet re-centring."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    best, best_var = None, -np.inf
    for _ in range(restarts):
        m = HoroPCA(dim=z.shape[1], n_components=2, lr=5e-2, max_steps=steps)
        m.fit(z, iterative=False, optim=True)
        var = -m.compute_loss(z, m.get_components()).item()
        if var > best_var:
            best, best_var = m, var
    return best


def project(model, z: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model.map_to_ball(z).numpy()


def exterior_angle(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Exterior angle at `x` of the triangle with the origin and `y`, in degrees."""
    s = math.sqrt(c)
    a, b = s * x.norm(dim=-1), s * y.norm(dim=-1)
    cos = (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp(min=1e-30)
    dist = torch.acosh(
        (torch.cosh(a) * torch.cosh(b) - torch.sinh(a) * torch.sinh(b) * cos).clamp(
            min=1 + 1e-12
        )
    )
    interior = torch.acos(
        (
            (torch.cosh(a) * torch.cosh(dist) - torch.cosh(b))
            / (torch.sinh(a) * torch.sinh(dist)).clamp(min=1e-30)
        ).clamp(-1, 1)
    )
    return torch.rad2deg(math.pi - interior)


def fit_plane(HoroPCA, background: torch.Tensor, c: float):
    """Fit the plane and rotate it so that the background texts lie on the right."""
    zb = to_ball(background, c)
    model = fit(HoroPCA, zb)
    eb = project(model, zb)
    cen = eb.mean(0)
    a = -math.atan2(cen[1], cen[0])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    eb = eb @ R.T
    kept = float(
        ball_distance(torch.from_numpy(eb)).pow(2).mean() / ball_distance(zb).pow(2).mean()
    )
    return model, R, np.round(eb, 5), kept


def tree_entry(tree, V, e, c):
    """Positions and exact measurements of one tree. `V` are its tangent vectors."""
    pa = np.array([n["parent"] for n in tree["nodes"][1:]])
    ch = np.arange(1, len(tree["nodes"]))
    rad = np.linalg.norm(e, axis=1)
    depth = (math.sqrt(c) * V.norm(dim=-1)).numpy()
    ext = exterior_angle(V[pa], V[ch], c)
    return {
        "noun": tree["noun"],
        "xy": e.tolist(),
        "steps": len(ch),
        "outward_in_plane": int((rad[ch] > rad[pa]).sum()),
        "in_front": int((ext < 90).sum()),
        "deeper_true": int((depth[ch] > depth[pa]).sum()),
        "exterior_angle_deg": ext.tolist(),
        "true_depth": depth.tolist(),
        "plane_depth": (2 * np.arctanh(rad)).tolist(),
    }


def single_hyperboloid(HoroPCA, trees, T, B, c):
    s = math.sqrt(c)
    model, R, eb, kept = fit_plane(HoroPCA, B, c)
    res = {
        "curvature": c,
        "background_xy": eb.tolist(),
        "variance_kept": kept,
        "background_true_depth": [
            float(v) for v in np.quantile((s * B.norm(dim=-1)).numpy(), [0.05, 0.5, 0.95])
        ],
        "background_plane_depth": [
            float(v)
            for v in np.quantile(2 * np.arctanh(np.linalg.norm(eb, axis=1)), [0.05, 0.5, 0.95])
        ],
        "trees": [],
    }
    for tree, V in zip(trees, T):
        e = project(model, to_ball(V, c)) @ R.T
        res["trees"].append(tree_entry(tree, V, e, c))
    return res


def product(HoroPCA, trees, T, B, curv):
    planes = {}
    res = {
        "curvature": float(curv[0]),
        "trees": [],
        "factors": T[0].shape[1],
        "factor_dim": T[0].shape[2],
    }
    for tree, V in zip(trees, T):
        pa = np.array([n["parent"] for n in tree["nodes"][1:]])
        ch = np.arange(1, len(tree["nodes"]))
        depth_all = curv.sqrt()[None] * V.norm(dim=-1)
        f = int(depth_all.mean(0).argmax())
        c = float(curv[f])
        if f not in planes:
            planes[f] = fit_plane(HoroPCA, B[:, f], c)
        model, R, eb, kept = planes[f]
        e = project(model, to_ball(V[:, f], c)) @ R.T
        entry = tree_entry(tree, V[:, f], e, c)
        entry.update(
            factor=f,
            background_xy=eb.tolist(),
            variance_kept=kept,
            in_front_mean_over_factors=float(
                np.mean(
                    [
                        (exterior_angle(V[pa, g], V[ch, g], float(curv[g])) < 90).sum().item()
                        for g in range(V.shape[1])
                    ]
                )
            ),
        )
        res["trees"].append(entry)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horopca-dir", required=True)
    parser.add_argument("--embeddings", nargs="+", required=True, metavar="NAME=FILE")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    HoroPCA = load_horopca(args.horopca_dir)
    trees = json.load(open(TREES))["trees"]
    background = json.load(open(BACKGROUND))
    out = {"trees": trees, "models": {}}

    for spec in args.embeddings:
        name, path = spec.split("=", 1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        ix = {t: i for i, t in enumerate(d["texts"])}
        text = d["text"].double()
        B = text[[ix[t] for t in background]]
        T = [text[[ix[n["text"]] for n in tree["nodes"]]] for tree in trees]
        if text.dim() == 3:
            res = product(HoroPCA, trees, T, B, d["curvature"].double())
        else:
            res = single_hyperboloid(HoroPCA, trees, T, B, float(d["curvature"]))
        out["models"][name] = res
        steps = sum(t["steps"] for t in res["trees"])
        print(
            f"{name}: child in front of its parent in "
            f"{sum(t['in_front'] for t in res['trees'])}/{steps} steps; sqrt(c) d(x, 0) from "
            f"{min(min(t['true_depth']) for t in res['trees']):.2f} to "
            f"{max(max(t['true_depth']) for t in res['trees']):.2f}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
