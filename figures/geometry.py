"""Geometry figure: one column per model, three rows (text width).

    (a) learned curvature against iteration, with the floor c_min
    (b) kernel density of the distance from the origin, on one axis shared by all panels
    (c) distance distortion r_d for text-image pairs at the converged curvature

    python figures/geometry.py

Source: results/geometry_vit_b.json, written by scripts/export_geometry.py. Densities are on each signal's own
support (Scott's rule); r_d is drawn over a per-panel r_2 window padded by 0.2 around the 1st-99th percentile box,
on one colour scale from 1.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import style

MODELS = {"hycoclip": "HyCoCLIP", "argent": "ARGENT", "phyclip": "PHyCLIP", "gromovclip": "GromovCLIP (ours)"}
# Modalities: one hue per modality, the box signal a lighter step of it; blue and orange stay the methods'.
SIGNALS = {"image": ("Image", "#117733"), "box_image": ("Box image", "#44BB99"),
           "text": ("Text", "#882255"), "box_text": ("Box text", "#CC6677")}
# r_d on a diverging blue-orange scale.
RD_CMAP = LinearSegmentedColormap.from_list(
    "rd_div", ["#1c5cab", "#2a78d6", "#86b6ef", "#f0efec", "#f5b483", "#eb6834", "#c2461a"])
# Box signals first, so the full signal's darker line stays visible where the two overlap.
DRAW_ORDER = ["box_image", "image", "box_text", "text"]
RES, PAD = 300, 0.2


def r_d(r1, r2, c, theta):
    """Distance distortion: geodesic distance after the exponential map over the Euclidean distance before it."""
    s = np.sqrt(c)
    a, b = s * r1, s * r2
    arg = np.cosh(a) * np.cosh(b) - np.sinh(a) * np.sinh(b) * np.cos(theta)
    return np.arccosh(np.maximum(arg, 1.0 + 1e-15)) / np.sqrt(np.maximum(a**2 + b**2 - 2 * a * b * np.cos(theta), 1e-15))


def kde(d, n=800):
    """scipy.stats.gaussian_kde with Scott's rule, evaluated on the signal's own support."""
    factor = len(d) ** (-1 / 5)
    h = factor * d.std(ddof=1)
    pad = 4 * factor * d.std()
    x = np.linspace(d.min() - pad, d.max() + pad, n)
    y = np.exp(-0.5 * ((x[:, None] - d[None, :]) / h) ** 2).sum(1) / (len(d) * h * np.sqrt(2 * np.pi))
    return x, y


def tidy(ax):
    ax.grid(color=style.EDGE_GRAY, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    style.apply()
    data = json.load(open(style.RESULTS / "geometry_vit_b.json"))["models"]

    fig = plt.figure(figsize=(style.TEXT_IN, 3.95))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.07], height_ratios=[1, 0.9, 1.1],
                          hspace=0.62, wspace=0.34, left=0.075, right=0.93, top=0.94, bottom=0.085)
    rows = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)]

    # (a) curvature
    last = max(d["curvature"]["steps"][-1] for d in data.values())
    for ax, (key, name) in zip(rows[0], MODELS.items()):
        cur = data[key]["curvature"]
        steps, curvs = np.array(cur["steps"]), np.array(cur["curvs"])
        colour = style.GROMOV if key.startswith("gromov") else style.REST
        many = len(curvs) > 1
        for f in curvs:
            ax.plot(steps, f, color=colour, lw=0.35 if many else 1.2, alpha=0.5 if many else 1.0)
        ax.axhline(cur["floor"], color=style.LABEL_TEXT, lw=0.6, ls=(0, (3, 2)), alpha=0.7)
        ax.text(0.97, 0.93, f"$c = {data[key]['c_final']:.2f}$", transform=ax.transAxes, ha="right", va="top",
                fontsize=7)
        ax.set_title(name, pad=3)
        ax.set_xscale("log")
        ax.set_xlim(100, last)
        ax.set_xticks([1e3, 1e4, 1e5], ["1k", "10k", "100k"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.5)
        tidy(ax)
        if ax is not rows[0][0]:
            ax.tick_params(labelleft=False)
    floor = data["hycoclip"]["curvature"]["floor"]
    rows[0][0].set_yticks([floor, 0.5, 1.0, 1.5], [r"$c_{\min}$", "0.5", "1", "1.5"])
    rows[0][0].set_ylabel("Curvature $c$")
    fig.text(0.5 * (0.075 + 0.93) - 0.03, 0.665, "Iteration", ha="center", fontsize=8)

    # (b) radial distributions, one shared distance axis
    hi = max(np.percentile(np.array(d), 99.9) for m in data.values() for d in m["radii"].values())
    hi = np.ceil(hi * 1.04 * 10) / 10
    for ax, key in zip(rows[1], MODELS):
        for sig in DRAW_ORDER:
            label, colour = SIGNALS[sig]
            x, y = kde(np.array(data[key]["radii"][sig]))
            ax.fill_between(x, y, color=colour, alpha=0.15, lw=0)
            ax.plot(x, y, color=colour, lw=0.9, label=label)
        ax.set_xlim(0, hi)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        tidy(ax)
    rows[1][0].set_ylabel("Density")
    handles, labels = rows[1][0].get_legend_handles_labels()
    order = [labels.index(SIGNALS[s][0]) for s in SIGNALS]
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    leg_ax = fig.add_subplot(gs[1, 4])
    leg_ax.axis("off")
    leg_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=6.5, handlelength=1.2,
                  borderaxespad=0, bbox_to_anchor=(-0.4, 0.5))
    fig.text(0.5 * (0.075 + 0.93) - 0.03, 0.355, "Distance from origin", ha="center", fontsize=8)

    # (c) distance distortion, per-panel r_2 window, one colour scale
    theta = np.linspace(0, np.pi, RES)
    surf, grids = {}, {}
    for key in MODELS:
        p = data[key]["distortion"]
        r2 = np.linspace(max(0.0, p["y_lo"] - PAD), p["y_hi"] + PAD, RES)
        grids[key] = np.meshgrid(theta, r2)
        surf[key] = r_d(p["r1"], grids[key][1], p["c"], grids[key][0])
    vmax = max(s.max() for s in surf.values())
    for ax, key in zip(rows[2], MODELS):
        p = data[key]["distortion"]
        T, R2 = grids[key]
        im = ax.pcolormesh(T, R2, surf[key], cmap=RD_CMAP, vmin=1.0, vmax=vmax, shading="auto", rasterized=True)
        ax.add_patch(Rectangle((p["th_lo"], p["y_lo"]), p["th_hi"] - p["th_lo"], p["y_hi"] - p["y_lo"], fill=False,
                               ec="black", lw=0.8, ls=(0, (2, 1.5))))
        ax.text(0.96, 0.06, f"$r_1 = {p['r1']:.2f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=style.EDGE_GRAY, lw=0.4))
        ax.set_xticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
        ax.set_xlabel(r"$\theta$", labelpad=1)
        inside = surf[key][(R2 >= p["y_lo"]) & (R2 <= p["y_hi"]) & (T >= p["th_lo"]) & (T <= p["th_hi"])]
        print(f"{key:17s} c={p['c']:.3f} r1={p['r1']:.3f} r_d box [{inside.min():.3f}, {inside.max():.3f}]"
              f" grid max {surf[key].max():.3f}")
    rows[2][0].set_ylabel("Image radius $r_2$")
    cax = fig.add_subplot(gs[2, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$r_d$", labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)

    for r, letter in enumerate("abc"):
        box = rows[r][0].get_position()
        fig.text(0.005, box.y1 + 0.01, f"({letter})", fontsize=8, fontweight="bold", va="bottom")

    style.save(fig, "geometry_vit_b.pdf")

if __name__ == "__main__":
    main()
