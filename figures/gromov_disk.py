"""Power-ratio Gromov loss (1 - r)^k over every child position on the Poincare disk.

Same disk, curvature and parent as cone_disk.py, so the two figures compare directly.
Panels: k = 1 and k = 2. The loss is zero only on the geodesic ray beyond the parent.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import style
from cone_disk import C, PARENT_DIST, N, VIEW_FRAC, d_hyp, radius_at_distance

KS = [1, 2]
LEVELS = [0.01, 0.05, 0.1, 0.25, 0.5]


def gromov_ratio(q, x, c):
    origin = np.zeros(2)
    dx0 = d_hyp(x, origin, c)
    prod = 0.5 * (dx0 + d_hyp(q, origin, c) - d_hyp(x, q, c))
    return np.clip(prod / dx0, 0.0, 1.0)


def main():
    style.apply()
    rho = radius_at_distance(PARENT_DIST, C)
    dom = rho / VIEW_FRAC  # radius of the zoomed view, not of the disk
    x = np.array([rho, rho]) / np.sqrt(2)

    g = np.linspace(-dom, dom, N)
    gx, gy = np.meshgrid(g, g)
    q = np.stack([gx, gy], axis=-1)
    inside = gx ** 2 + gy ** 2 < (0.995 * dom) ** 2
    r = gromov_ratio(q, x, C)

    fig, axes = plt.subplots(1, 2, figsize=(style.COLUMN_IN, 1.75))
    for ax, k in zip(axes, KS):
        loss = np.where(inside, (1 - r) ** k, np.nan)
        im = ax.imshow(loss, extent=[-dom, dom, -dom, dom], origin="lower",
                       cmap=style.GROMOV_CMAP, vmin=0, vmax=1, interpolation="bilinear")
        ax.contour(gx, gy, loss, levels=LEVELS, colors=style.NODE_STROKE, linewidths=0.35)
        # Zero set: the geodesic ray from the parent away from the origin (a straight
        # diameter segment in the Poincare disk, since it passes through the origin).
        ax.plot([x[0], dom * 0.995 / np.sqrt(2)], [x[1], dom * 0.995 / np.sqrt(2)],
                color=style.LABEL_TEXT, lw=0.9)
        ax.add_patch(Circle((0, 0), dom, fill=False, ec=style.NODE_STROKE, lw=0.6, ls=(0, (3, 2))))  # dashed: crop, not the disk boundary
        ax.plot(0, 0, "o", ms=4, mfc=style.ROOT_FILL, mec=style.NODE_STROKE, mew=0.6)
        ax.plot(*x, "o", ms=4, mfc=style.MARK_FILL, mec=style.NODE_STROKE, mew=0.6)
        ax.annotate(r"$\mathbf{0}$", (0, 0), xytext=(-3, -9), textcoords="offset points", fontsize=7)
        ax.annotate(r"$\boldsymbol{x}$", x, xytext=(-9, 2), textcoords="offset points", fontsize=7)
        ax.set_title(rf"$k = {k}$", fontsize=8, pad=2)
        ax.set_xlim(-dom * 1.02, dom * 1.02)
        ax.set_ylim(-dom * 1.02, dom * 1.02)
        ax.set_aspect("equal")
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(width=0.4, length=2)
    cbar.set_label(r"$\ell_{\mathrm{Gromov}}$", labelpad=1)
    style.save(fig, "gromov_disk.pdf")


if __name__ == "__main__":
    main()
