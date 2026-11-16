"""Gradient fields of each hierarchy loss around a fixed parent, as the optimizer sees it (single column).

The models train the encoder outputs with AdamW: a child is y = expm_0(v), and the hierarchy term is
differentiated with respect to the tangent vector v, without any Riemannian correction. This script
does the same. Layout, top to bottom:
  the Poincare disk with the parent x and the window that the rows below zoom into;
  then per loss, direction: where a small plain gradient step on v moves y in the disk (unit arrows; none where the
             gradient vanishes), one panel per loss;
  magnitude: ||grad_v|| of the weighted term; the angular losses share a linear scale clipped at ANG_MAX
             (their gradient diverges at the parent, marked infinity), the Gromov loss has its own.
One row per loss, direction and magnitude side by side.
Losses, with the weights used in training (see the paper's supplementary material):
  entailment cone, eta = 0.7 as on cross-modal pairs, K = 0.1, weight 0.1 (HyCoCLIP, PHyCLIP);
  ARGENT, Huber(ext(x, y), beta = pi/4) with h = 1 as on cross-modal pairs, weight 0.1;
  Gromov loss (1 - r)^k with k = 2 as on cross-modal pairs, weight 0.5.
Parent as in cone_disk.py: c = 0.1, d(x, 0) = 0.6, in the saturated regime.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import ConnectionPatch

import style

C, K, PARENT_DIST, PHI = 0.1, 0.1, 0.6, np.pi / 4
ETA, BETA, K_GROMOV = 0.7, np.pi / 4, 2
WEIGHTS = {"cone": 0.1, "argent": 0.1, "gromov": 0.5}   # per relation, as in training
HALF = 0.55          # half-width of the window, as a fraction of the parent's Poincare radius
N_MAG, N_DIR = 400, 120
H = 1e-6             # finite-difference step in v
SQC = np.sqrt(C)


def lift(v):
    """Tangent vector at the origin (..., 2) -> hyperboloid point (space (..., 2), time (...))."""
    n = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)
    space = np.sinh(SQC * n) / SQC * v / n
    return space, np.cosh(SQC * n[..., 0]) / SQC


def inner(a, b):
    return np.sum(a[0] * b[0], axis=-1) - a[1] * b[1]


def dist(a, b):
    return np.arccosh(np.maximum(-C * inner(a, b), 1.0)) / SQC


def ext(x, y):
    """Exterior angle at the parent x (the exterior angle of the paper's background section)."""
    cxy = C * inner(x, y)
    num = y[1] + x[1] * cxy
    den = np.linalg.norm(x[0], axis=-1) * np.sqrt(np.maximum(cxy ** 2 - 1, 1e-30))
    return np.arccos(np.clip(num / den, -1, 1))


def losses(x, v):
    y = lift(v)
    e = ext(x, y)
    aper = np.arcsin(np.minimum(2 * K / (SQC * np.linalg.norm(x[0], axis=-1)), 1.0))
    huber = np.where(e <= BETA, 0.5 * e ** 2, BETA * (e - 0.5 * BETA))
    o = (np.zeros_like(x[0]), np.full_like(x[1], 1 / SQC))
    dx0 = dist(x, o)
    r = 0.5 * (dx0 + dist(y, (np.zeros_like(y[0]), np.full_like(y[1], 1 / SQC))) - dist(x, y)) / dx0
    return {"cone": WEIGHTS["cone"] * np.maximum(0, e - ETA * aper),
            "argent": WEIGHTS["argent"] * huber,
            "gromov": WEIGHTS["gromov"] * (1 - np.clip(r, 0, 1)) ** K_GROMOV}


def to_disk(v):
    """Tangent vector at the origin -> Poincare-disk point (radius 1/sqrt(c))."""
    n = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)
    return np.tanh(SQC * n / 2) / SQC * v / n


def from_disk(p):
    n = np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-15)
    return 2 / SQC * np.arctanh(np.minimum(SQC * n, 1 - 1e-12)) * p / n


def grads(x, v):
    """Euclidean gradient of every loss with respect to v, by central differences."""
    out = {}
    base = losses(x, v)
    for key in base:
        out[key] = np.zeros(v.shape)
    for i in range(2):
        e = np.zeros(2)
        e[i] = H
        lp, lm = losses(x, v + e), losses(x, v - e)
        for key in base:
            out[key][..., i] = (lp[key] - lm[key]) / (2 * H)
    return base, out


def window(xp, n):
    w = HALF * np.linalg.norm(xp)
    # Shift rows by a fraction of a cell so no sample falls exactly on the ray through x, where the
    # exterior angle equals pi and has no derivative.
    off = 0.37 * 2 * w / (n - 1)
    gx, gy = np.meshgrid(np.linspace(xp[0] - w, xp[0] + w, n), np.linspace(xp[1] - w, xp[1] + w, n) + off)
    return gx, gy, np.stack([gx, gy], axis=-1), w


KEYS = ["cone", "argent", "gromov"]
TITLES = {"cone": "Entailment cone", "argent": "ARGENT", "gromov": "Gromov"}
N_ARROW = 11
ANG_MAX = 3.0      # clip of the shared angular scale; their gradient diverges at the parent
ZOOM = style.LABEL_TEXT   # zoom window in the overview and the frames of the zoomed panels


def parent_marker(ax, xp, label=True):
    ax.plot(*xp, "o", ms=4, mfc=style.GROMOV, mec=style.LABEL_TEXT, mew=0.6, zorder=6)
    if label:
        ax.annotate(r"$\boldsymbol{x}$", xp, xytext=(3, -8), textcoords="offset points", fontsize=7)


def zoomed(ax, xp, w):
    ray = np.array([np.cos(PHI), np.sin(PHI)])
    ax.plot([xp[0] - 2 * w * ray[0], xp[0] + 2 * w * ray[0]], [xp[1] - 2 * w * ray[1], xp[1] + 2 * w * ray[1]],
            color=style.LABEL_TEXT, lw=0.5, ls=(0, (3, 2)))   # the parent's ray from the origin
    parent_marker(ax, xp)
    ax.set_xlim(xp[0] - w, xp[0] + w)
    ax.set_ylim(xp[1] - w, xp[1] + w)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():   # framed in the colour of the zoom window in the overview
        sp.set_linewidth(0.8)
        sp.set_color(ZOOM)


def main():
    style.apply()
    vx = PARENT_DIST * np.array([np.cos(PHI), np.sin(PHI)])
    xp = to_disk(vx)
    x = lift(vx)
    xb = (x[0][None, None], np.array(x[1])[None, None])

    gx, gy, p, w = window(xp, N_MAG)
    base, g = grads(xb, from_disk(p))
    ax_, ay_, pa, _ = window(xp, N_ARROW)
    va = from_disk(pa)
    _, ga = grads(xb, va)

    # Two linear scales: the angular losses share one, clipped at ANG_MAX, since their gradient diverges
    # at the parent (marked infinity); the Gromov gradient is bounded and gets its own full range.
    mags = {k: np.linalg.norm(g[k], axis=-1) for k in KEYS}
    gromov_max = float(np.nanmax(mags["gromov"]))
    norms = {"cone": Normalize(0, ANG_MAX), "argent": Normalize(0, ANG_MAX), "gromov": Normalize(0, gromov_max)}
    levels = {"cone": [0.5, 1, 2], "argent": [0.5, 1, 2], "gromov": [1, 2, 3, 4]}
    cmap = style.GRAD_CMAP.copy()
    cmap.set_bad(style.EDGE_GRAY)      # zero gradient: the cone's dead zone
    ims = {}

    fig = plt.figure(figsize=(style.COLUMN_IN, 4.9))
    gs = fig.add_gridspec(4, 3, height_ratios=[0.95, 1, 1, 1], width_ratios=[1, 1, 0.06],
                          hspace=0.1, wspace=0.05)

    # Overview: the whole disk, the parent on its ray, and the window the rows below zoom into.
    ov = fig.add_subplot(gs[0, :2])
    R = 1 / SQC
    ray = np.array([np.cos(PHI), np.sin(PHI)])
    ov.add_patch(plt.Circle((0, 0), R, fill=False, ec=style.NODE_STROKE, lw=0.6))
    ov.plot([0, R * ray[0]], [0, R * ray[1]], color=style.LABEL_TEXT, lw=0.5, ls=(0, (3, 2)))
    ov.plot(0, 0, "o", ms=3.5, mfc=style.ROOT_FILL, mec=style.NODE_STROKE, mew=0.5, zorder=5)
    ov.annotate(r"$\mathbf{0}$", (0, 0), xytext=(-3, -9), textcoords="offset points", fontsize=7)
    ov.plot(*xp, "o", ms=2, mfc=style.GROMOV, mec="none", zorder=6)
    ov.add_patch(plt.Rectangle((xp[0] - w, xp[1] - w), 2 * w, 2 * w, fill=False, ec=ZOOM, lw=0.9, zorder=7))
    ov.annotate(r"$\boldsymbol{x}$", (xp[0] + w, xp[1]), xytext=(2, -2), textcoords="offset points", fontsize=7)
    ov.set_xlim(-R * 1.02, R * 1.02)
    ov.set_ylim(-R * 1.02, R * 1.02)
    ov.set_aspect("equal")
    ov.axis("off")

    for i, key in enumerate(KEYS):
        colour = style.GROMOV if key == "gromov" else style.REST   # fixed colour per method
        dead = base[key] <= 1e-12

        # Direction: where a plain gradient step on v moves the child in the disk.
        ax = fig.add_subplot(gs[i + 1, 0])
        if dead.any():
            ax.contourf(gx, gy, dead.astype(float), levels=[0.5, 1.5], colors=[style.EDGE_GRAY], alpha=0.5)
        step = to_disk(va - 1e-4 * ga[key]) - pa
        sn = np.linalg.norm(step, axis=-1)
        live = (sn > 1e-13) & (np.linalg.norm(pa - xp, axis=-1) > 0.06 * w)
        u = np.where(live, step[..., 0] / np.maximum(sn, 1e-30), np.nan)
        vv = np.where(live, step[..., 1] / np.maximum(sn, 1e-30), np.nan)
        ax.quiver(ax_, ay_, u, vv, color=colour, angles="xy", scale_units="xy", scale=N_ARROW / (1.5 * w),
                  width=0.009, headwidth=3.8, headlength=3.8, headaxislength=3.4, zorder=3)
        zoomed(ax, xp, w)
        ax.set_ylabel(TITLES[key], fontsize=7.5, labelpad=3, color=colour)
        if i == 0:
            ax.set_title("Direction", fontsize=7.5, pad=3)

        # Magnitude.
        ax = fig.add_subplot(gs[i + 1, 1])
        m = np.where(dead, np.nan, mags[key])
        ims[key] = ax.imshow(m, extent=[gx.min(), gx.max(), gy.min(), gy.max()], origin="lower",
                             cmap=cmap, norm=norms[key], interpolation="bilinear")
        ax.contour(gx, gy, m, levels=levels[key], colors=style.LABEL_TEXT, linewidths=0.35,
                   linestyles=[(0, (2.5, 1.5))])
        if key != "gromov":
            ax.annotate(r"$\infty$", xp, xytext=(-9, 3), textcoords="offset points", fontsize=7.5,
                        color=style.LABEL_TEXT)
        zoomed(ax, xp, w)
        if i == 0:
            ax.set_title("Magnitude", fontsize=7.5, pad=3)

    for rows, key, ticks, extend in [(slice(1, 3), "cone", [0, 1, 2, 3], "max"),
                                     (slice(3, 4), "gromov", [0, 2, 4], "neither")]:
        cbar = fig.colorbar(ims[key], cax=fig.add_subplot(gs[rows, 2]), ticks=ticks, extend=extend,
                            extendfrac=0.06)
        if extend == "max":
            cbar.ax.text(0.5, 1.09, r"$\infty$", transform=cbar.ax.transAxes, ha="center", va="bottom",
                         fontsize=7)
        cbar.outline.set_linewidth(0.4)
        cbar.ax.tick_params(width=0.4, length=2, labelsize=6)
        cbar.set_label(r"$\|\nabla_{\boldsymbol{v}}\,\ell\|$", fontsize=7, labelpad=1)
    style.save(fig, "grad_flow.pdf")
    print("angular clip", ANG_MAX, "Gromov max", round(gromov_max, 3))


if __name__ == "__main__":
    main()
