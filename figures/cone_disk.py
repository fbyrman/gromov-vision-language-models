"""Entailment-cone loss over every child position on the Poincare disk.

Figure 1 (top) of the paper: c = 0.1, K = 0.1,
parent at hyperbolic distance 0.6 from the origin, where the aperture is saturated.
Panels: eta = 1 (cone opens to a half-space) and eta = 1.2 (cone folds backwards).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import style

C, K, PARENT_DIST = 0.1, 0.1, 0.6  # sqrt(c) * ||x_space|| = 0.19 <= 2K: saturated
ETAS = [1.0, 1.2]
N = 900
# A saturated parent always sits within ~10% of the disk radius (tanh(sqrt(c) d / 2) <= 0.1),
# so the figures zoom into the central region, placing x at VIEW_FRAC of the view radius.
VIEW_FRAC = 0.4


def d_hyp(u, v, c):
    """Geodesic distance on the Poincare ball of curvature -c (arrays of shape (..., 2))."""
    diff2 = np.sum((u - v) ** 2, axis=-1)
    ru, rv = np.sum(u ** 2, axis=-1), np.sum(v ** 2, axis=-1)
    denom = np.maximum((1 - c * ru) * (1 - c * rv), 1e-12)
    return np.arccosh(np.maximum(1 + 2 * c * diff2 / denom, 1.0)) / np.sqrt(c)


def radius_at_distance(dist, c):
    a = np.cosh(dist * np.sqrt(c))
    return np.sqrt((a - 1) / (c * (a + 1)))


def cone_loss(q, x, c, eta):
    s = np.sqrt(c)
    origin = np.zeros(2)
    a = d_hyp(x, origin, c) * s
    b = d_hyp(x, q, c) * s
    o = d_hyp(q, origin, c) * s
    cos_angle = (np.cosh(a) * np.cosh(b) - np.cosh(o)) / np.maximum(np.sinh(a) * np.sinh(b), 1e-12)
    ext = np.pi - np.arccos(np.clip(cos_angle, -1, 1))
    arg = 2 * K / np.sinh(a)
    aper = np.pi / 2 if arg >= 1 else np.arcsin(arg)
    return np.maximum(0.0, ext - eta * aper)


def main():
    style.apply()
    rho = radius_at_distance(PARENT_DIST, C)
    dom = rho / VIEW_FRAC  # radius of the zoomed view, not of the disk
    x = np.array([rho, rho]) / np.sqrt(2)

    g = np.linspace(-dom, dom, N)
    gx, gy = np.meshgrid(g, g)
    q = np.stack([gx, gy], axis=-1)
    inside = gx ** 2 + gy ** 2 < (0.995 * dom) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN_IN, 1.75))
    for ax, eta in zip(axes, ETAS):
        loss = np.where(inside, cone_loss(q, x, C, eta), np.nan)
        im = ax.imshow(loss, extent=[-dom, dom, -dom, dom], origin="lower",
                       cmap=style.REST_CMAP, vmin=0, vmax=np.pi / 2, interpolation="bilinear")
        zero = np.where(inside, (loss <= 1e-9).astype(float), np.nan)
        ax.contourf(gx, gy, zero, levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.contour(gx, gy, zero, levels=[0.5], colors=style.NODE_STROKE, linewidths=0.6)
        ax.add_patch(Circle((0, 0), dom, fill=False, ec=style.NODE_STROKE, lw=0.6, ls=(0, (3, 2))))  # dashed: crop, not the disk boundary
        ax.plot(0, 0, "o", ms=4, mfc=style.ROOT_FILL, mec=style.NODE_STROKE, mew=0.6)
        ax.plot(*x, "o", ms=4, mfc=style.MARK_FILL, mec=style.NODE_STROKE, mew=0.6)
        ax.annotate(r"$\mathbf{0}$", (0, 0), xytext=(-3, -9), textcoords="offset points", fontsize=7)
        ax.annotate(r"$\boldsymbol{x}$", x, xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(rf"$\eta = {eta:g}$", fontsize=8, pad=2)
        ax.set_xlim(-dom * 1.02, dom * 1.02)
        ax.set_ylim(-dom * 1.02, dom * 1.02)
        ax.set_aspect("equal")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, ticks=[0, np.pi / 4, np.pi / 2])
    cbar.ax.set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$"])
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(width=0.4, length=2)
    cbar.set_label(r"$\ell_{\mathrm{entail}}$", labelpad=1)
    style.save(fig, "cone_disk.pdf")


if __name__ == "__main__":
    main()
