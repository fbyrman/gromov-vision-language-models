"""One text tree in the text plane of each model (column width): Figure 5 of the paper.

GromovCLIP, HyCoCLIP, ARGENT and PHyCLIP, all ViT-B/16 on CC3M. The plane of a model is fitted by HoroPCA on the
1,500 background texts (grey), not on the tree; PHyCLIP draws the tree in its own factor. The four panels, two by two,
magnify the tree. Each panel has its whole Poincare disk outside it, above the top row and below the bottom row, with
the origin and a box around the magnified region; dashed lines join the box to the panel.

The two branches of the tree take their own colour and marker (red circles, blue triangles) and the strength of the
colour gives the level: light at level 2, strong at level 4; the root is grey. A step is solid when, in the full
embedding space, the child lies in front of its parent (exterior angle at the parent below 90 degrees), and dashed
otherwise. Each panel gives its model and the number of solid steps (--key angle: the mean exterior angle instead).
The tree itself is written out above the figure by text_trees_tikz.py, so the key below names only the line styles
(--names adds the root and the branches).

    python figures/text_tree_plane.py boat --no-key

Source: results/text_trees.json (trees in assets/qualitative/text_trees.json).
"""
import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

import style
from tree_plot import BACKGROUND, DASH, MARKER, MODELS, MS, RAMP, ROOT, ROOT_MS, branch_style, geodesic, origin_mark, window

SHOWN = ["gromovclip", "hycoclip", "argent", "phyclip"]
PANEL_H, HGAP, DISK, DGAP, VGAP, TOP = 0.90, 0.08, 0.78, 0.07, 0.08, 0.02      # inches
FRAME = "black"
KEY_ROW, FS_KEY, FS_SCORE, FS_NAME = 0.135, 6.5, 5.6, 6.8
LW, LW_DASH = 0.9, 1.1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("noun", nargs="?", default="boat")
    p.add_argument("--data", default=str(style.RESULTS / "text_trees.json"))
    p.add_argument("--tag", default="")
    p.add_argument("--key", choices=["count", "angle"], default="count", help="number in each panel's key")
    p.add_argument("--names", action="store_true", help="a key row that names the root and the branches")
    p.add_argument("--no-key", action="store_true", help="no key row below the figure (the caption names the line styles)")
    a = p.parse_args()

    style.apply()
    data = json.load(open(a.data))
    n = [t["noun"] for t in data["trees"]].index(a.noun)
    tree = data["trees"][n]
    nodes = tree["nodes"]
    level, branch, colour, marker = branch_style(tree)
    names = [nodes[j]["text"].removeprefix("a ").removeprefix("an ") for j in np.flatnonzero(level <= 2)]
    th = np.linspace(-math.pi, math.pi, 2000)

    W = style.COLUMN_IN
    pw = (W - HGAP) / 2
    key_h = 0.0 if a.no_key else (2 if a.names else 1) * KEY_ROW + 0.06
    H = TOP + DISK + DGAP + PANEL_H + VGAP + PANEL_H + DGAP + DISK + key_h
    fig = plt.figure(figsize=(W, H))
    any_inward = False
    for i, key in enumerate(SHOWN):
        row, col = divmod(i, 2)
        title = MODELS[key]
        res = data["models"][key]["trees"][n]
        bg = np.array(res.get("background_xy") or data["models"][key]["background_xy"])   # PHyCLIP: the plane of the tree's factor
        xy = np.array(res["xy"])
        rad = np.linalg.norm(xy, axis=1)
        ext = np.array(res["exterior_angle_deg"])
        x0 = col * (pw + HGAP)
        if row == 0:                                                   # disk, panel
            y0 = H - TOP - DISK - DGAP - PANEL_H
            dy = H - TOP - DISK
        else:                                                          # panel, disk
            y0 = key_h + DISK + DGAP
            dy = key_h

        ax = fig.add_axes([x0 / W, y0 / H, pw / W, PANEL_H / H])
        xl, yl = window(xy, pw, PANEL_H)
        pad = 0.06 * (xl[1] - xl[0])
        xl, yl = [xl[0] - pad, xl[1] + pad], [yl[0] - pad * PANEL_H / pw, yl[1] + pad * PANEL_H / pw]
        ax.set_xlim(xl), ax.set_ylim(yl)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(FRAME), side.set_linewidth(0.7)
        span = 2 * np.arctanh(min(np.hypot(max(map(abs, xl)), max(map(abs, yl))), 0.999)) - 2 * np.arctanh(rad.min())
        step = next(s for s in (0.2, 0.1, 0.05, 0.02, 0.01) if span / s >= 3)
        for dist in np.arange(step, 4, step):
            r = math.tanh(dist / 2)
            ax.plot(r * np.cos(th), r * np.sin(th), color=style.EDGE_GRAY, lw=0.4, zorder=0)
        ax.scatter(bg[:, 0], bg[:, 1], s=1.0, color=BACKGROUND, lw=0, zorder=1)
        mx, my = 0.08 * (xl[1] - xl[0]), 0.14 * (yl[1] - yl[0])      # room for the label of the origin inside the frame
        if xl[0] + mx <= 0 <= xl[1] - mx and yl[0] + my <= 0 <= yl[1] - my:
            origin_mark(ax, ms=2.8, fontsize=6.5, offset=(-2, -2))
        front = [None] + [angle < 90 for angle in ext]                # exact, in the full space; per child node
        outward = 0
        for j, node in enumerate(nodes):
            if node["parent"] is None:
                continue
            g = geodesic(xy[node["parent"]], xy[j])
            outward += rad[j] > rad[node["parent"]]
            any_inward |= not front[j]
            ax.plot(g[:, 0], g[:, 1], color=colour[j], lw=LW if front[j] else LW_DASH, ls="-" if front[j] else DASH, zorder=3,
                    solid_capstyle="round", dash_capstyle="butt")
        for j in np.argsort(-level, kind="stable"):                   # deep levels first, so the root ends on top
            ax.plot(*xy[j], marker=marker[j][0], ms=marker[j][1], mfc=colour[j], mec="white", mew=0.35, ls="", zorder=9 - level[j])

        # The whole disk outside the panel, with the magnified region and lines to the panel's near corners.
        dx = fig.add_axes([(x0 + (pw - DISK) / 2) / W, dy / H, DISK / W, DISK / H])
        dx.add_patch(plt.Circle((0, 0), 1, fc="white", ec=style.NODE_STROKE, lw=0.6, zorder=1))
        dx.scatter(bg[:, 0], bg[:, 1], s=0.35, color=BACKGROUND, lw=0, zorder=2)
        origin_mark(dx, ms=2.6, fontsize=6.5, offset=(-2, 2) if row == 0 else (-2, -2), zorder=3)
        dx.add_patch(Rectangle((xl[0], yl[0]), xl[1] - xl[0], yl[1] - yl[0], fill=False, lw=0.7, color=FRAME, zorder=5))
        dx.set_xlim(-1.03, 1.03), dx.set_ylim(-1.03, 1.03)
        dx.set_aspect("equal")
        dx.axis("off")
        near = yl[0] if row == 0 else yl[1]
        for xc, xa in ((xl[0], 0), (xl[1], 1)):
            fig.add_artist(ConnectionPatch(xyA=(xc, near), coordsA=dx.transData, xyB=(xa, 1 if row == 0 else 0), coordsB=ax.transAxes,
                                           lw=0.45, color=FRAME, ls=(0, (2, 2)), zorder=5))
        zoom = 2 / (xl[1] - xl[0]) * pw / DISK
        fig.text((x0 + pw - (pw - DISK) / 4) / W, (dy + DISK / 2) / H, f"zoom\n$\\times{zoom:.0f}$", ha="center", va="center", ma="center",
                 fontsize=6.5, linespacing=1.2, color=style.NODE_STROKE)

        box = dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc="white", ec=style.EDGE_GRAY, lw=0.4, alpha=0.92)
        ax.text(0.03, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=FS_NAME, color="black", zorder=12, bbox=box)
        score = (f"in front  {res['in_front']}/{res['steps']}" if a.key == "count" else f"mean angle  {ext.mean():.0f}\u00b0")
        ax.text(0.97, 0.96, score, transform=ax.transAxes, ha="right", va="top", fontsize=FS_SCORE, color=style.LABEL_TEXT,
                zorder=12, bbox=box)
        print(f"{title:18s} {a.noun}: in front (exact) {res['in_front']}/{res['steps']}, mean exterior angle {ext.mean():.1f}, "
              f"child farther out in the plane {outward}/{res['steps']}, deeper in the full space {res['deeper_true']}/{res['steps']}; "
              f"magnification {zoom:.0f}; distance from the origin in the full space {min(res['true_depth']):.2f}-{max(res['true_depth']):.2f}"
              + (f"; factor {res['factor']}" if "factor" in res else ""))

    rend = fig.canvas.get_renderer()

    def put(x, y, text, **kw):
        t = fig.text(x / W, y / H, text, ha="left", va="center", fontsize=FS_KEY, color=style.LABEL_TEXT, **kw)
        fig.canvas.draw()
        return t.get_window_extent(rend).x1 / fig.dpi

    def mark(x, y, mk, ms, c):
        fig.add_artist(plt.Line2D([x / W], [y / H], marker=mk, ms=ms, mfc=c, mec="white", mew=0.35, ls="", transform=fig.transFigure))

    def line(x, y, **kw):
        fig.add_artist(plt.Line2D([x / W, (x + 0.2) / W], [y / H] * 2, color=style.NODE_STROKE, transform=fig.transFigure, **kw))
        return x + 0.24

    if a.no_key:
        style.save(fig, f"text_tree_plane_{a.noun}{'_' + a.tag if a.tag else ''}.pdf")
        return
    y = key_h - 0.06 - KEY_ROW / 2
    if a.names:                     # the root, then each branch with its three levels from light to strong
        x = 0.06
        mark(x + 0.04, y, "o", ROOT_MS, ROOT)
        x = put(x + 0.10, y, f"{names[0]} (1)") + 0.12
        for b in range(2):
            for s_ in range(3):
                mark(x + 0.04 + 0.075 * s_, y, MARKER[b], MS[b], RAMP[b][s_])
            x = put(x + 0.27, y, f"{names[1 + b]} (2–4)") + 0.12
        y -= KEY_ROW
    x = 0.06                        # the line styles and the background
    x = put(line(x, y, lw=LW, solid_capstyle="round"), y, "child in front") + 0.12
    if any_inward:
        x = put(line(x, y, lw=LW_DASH, ls=DASH), y, "not in front") + 0.12
    mark(x + 0.04, y, "o", 2.2, BACKGROUND)
    put(x + 0.10, y, "other texts")

    style.save(fig, f"text_tree_plane_{a.noun}{'_' + a.tag if a.tag else ''}.pdf")


if __name__ == "__main__":
    main()
