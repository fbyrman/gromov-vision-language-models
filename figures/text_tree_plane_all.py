"""All eight text trees in the text plane of each model (text width, two pages of four trees).

One row per tree: the tree written out on the left, then one panel per model (GromovCLIP, HyCoCLIP, ARGENT, PHyCLIP),
each zoomed on that tree in the model's plane. The planes are those of text_tree_plane.py (HoroPCA on the 1,500
background texts, grey), turned so that the background lies above the origin: a step upward is a step away from the
origin. There is no whole-disk inset here; Figure 5 shows where the trees sit in the disk.

Marks as in text_tree_plane.py (tree_plot.branch_style): the first branch red circles, the second blue triangles,
light at level 2 and strong at level 4, the root grey. A step is solid when, in the full embedding space, the child
lies in front of its parent (exterior angle below 90 degrees), dashed otherwise; the key in a panel counts the solid
steps. PHyCLIP draws each tree in the factor in which the tree lies farthest from the origin.

    python figures/text_tree_plane_all.py     # outputs/figures/text_tree_plane_all_1.pdf and _2.pdf

Source: results/text_trees.json (trees in assets/qualitative/text_trees.json).
"""
import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np

import style
from tree_plot import BACKGROUND, DASH, MODELS, branch_style, geodesic, origin_mark, window

SHOWN = ["gromovclip", "hycoclip", "argent", "phyclip"]
TREES_PER_PAGE = 4
OUTLINE, GAP, ROW_H, VGAP, TITLE, KEY = 1.70, 0.05, 1.80, 0.10, 0.17, 0.20      # inches
PITCH, INDENT, FS_TREE, FS_KEY, FS_SCORE = 0.0935, 0.12, 5.9, 6.5, 5.6
LW, LW_DASH = 0.9, 1.1
TURN = np.array([[0.0, -1.0], [1.0, 0.0]])          # 90 degrees: the background, on +x in the stored plane, goes up


def strip(text):
    return text[3:] if text.startswith("an ") else text[2:] if text.startswith("a ") else text


def outline(ax, tree, colour, marker, h):
    """The tree written out top to bottom, one line per text, indented by level, with the mark of each node."""
    nodes = tree["nodes"]
    top = h / 2 + (len(nodes) - 1) * PITCH / 2
    y = [top - j * PITCH for j in range(len(nodes))]                        # nodes are stored depth first
    x = [0.05 + INDENT * (n["level"] - 1) for n in nodes]
    for j, n in enumerate(nodes):
        if n["parent"] is not None:                                         # elbow from the parent's mark to the child's
            p = n["parent"]
            ax.plot([x[p], x[p], x[j] - 0.035], [y[p] - 0.03, y[j], y[j]], color=colour[j], lw=0.5, solid_joinstyle="miter", zorder=1)
    for j, n in enumerate(nodes):
        ax.plot(x[j], y[j], marker=marker[j][0], ms=marker[j][1] * 0.85, mfc=colour[j], mec="white", mew=0.3, ls="", zorder=3)
        text = "… " + n["tail"] if n["level"] == 4 else strip(n["text"])
        ax.text(x[j] + 0.055, y[j], text, ha="left", va="center", fontsize=FS_TREE, color=style.LABEL_TEXT, zorder=3)


def panel(ax, res, bg, tree, colour, marker, pw, ph):
    """One model's plane, zoomed on the tree. Returns whether a step is dashed."""
    nodes = tree["nodes"]
    level = np.array([n["level"] for n in nodes])
    xy = np.array(res["xy"]) @ TURN.T
    bg = bg @ TURN.T
    rad = np.linalg.norm(xy, axis=1)
    xl, yl = window(xy, pw, ph)
    pad = 0.08 * (yl[1] - yl[0])
    xl, yl = [xl[0] - pad * pw / ph, xl[1] + pad * pw / ph], [yl[0] - pad, yl[1] + pad]
    # Room for the key at the top or the bottom, whichever the tree leaves free: stretch the window there.
    f = (xy[:, 1] - yl[0]) / (yl[1] - yl[0])
    top_free = f.max() < 0.86
    if not top_free and f.min() < 0.14:
        grow = 0.16 * (yl[1] - yl[0])
        yl = [yl[0] - grow, yl[1]]
        cx, half = sum(xl) / 2, (yl[1] - yl[0]) * pw / ph / 2
        xl = [cx - half, cx + half]
    ax.set_xlim(xl), ax.set_ylim(yl)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(style.NODE_STROKE), side.set_linewidth(0.6)
    th = np.linspace(-math.pi, math.pi, 2000)
    span = 2 * np.arctanh(min(np.hypot(max(map(abs, xl)), max(map(abs, yl))), 0.999)) - 2 * np.arctanh(rad.min())
    step = next(s for s in (0.2, 0.1, 0.05, 0.02, 0.01) if span / s >= 3)
    for dist in np.arange(step, 4, step):
        r = math.tanh(dist / 2)
        ax.plot(r * np.cos(th), r * np.sin(th), color=style.EDGE_GRAY, lw=0.4, zorder=0)
    ax.scatter(bg[:, 0], bg[:, 1], s=1.0, color=BACKGROUND, lw=0, zorder=1)
    mx, my = 0.10 * (xl[1] - xl[0]), 0.08 * (yl[1] - yl[0])
    if xl[0] + mx <= 0 <= xl[1] - mx and yl[0] + my <= 0 <= yl[1] - my:
        origin_mark(ax, ms=2.8, fontsize=6.5, offset=(-2, -2))
    front = [None] + [a < 90 for a in res["exterior_angle_deg"]]
    for j, n in enumerate(nodes):
        if n["parent"] is None:
            continue
        g = geodesic(xy[n["parent"]], xy[j])
        ax.plot(g[:, 0], g[:, 1], color=colour[j], lw=LW if front[j] else LW_DASH, ls="-" if front[j] else DASH, zorder=3,
                solid_capstyle="round", dash_capstyle="butt")
    for j in np.argsort(-level, kind="stable"):
        ax.plot(*xy[j], marker=marker[j][0], ms=marker[j][1], mfc=colour[j], mec="white", mew=0.35, ls="", zorder=9 - level[j])
    # The key: the solid steps, in the free top or bottom corner farthest from the nodes.
    f = (xy - [xl[0], yl[0]]) / [xl[1] - xl[0], yl[1] - yl[0]] * [pw, ph]
    corners = [(c, r) for r in ((1,) if top_free else (0, 1)) for c in (0, 1)]
    far = lambda c: np.hypot(f[:, 0] - (0.32 if c[0] == 0 else pw - 0.32), f[:, 1] - (0.08 if c[1] == 0 else ph - 0.08)).min()
    c = max(corners, key=far)
    ax.text(0.04 if c[0] == 0 else 0.96, 0.025 if c[1] == 0 else 0.975, f"in front  {res['in_front']}/{res['steps']}",
            transform=ax.transAxes, ha="left" if c[0] == 0 else "right", va="bottom" if c[1] == 0 else "top", fontsize=FS_SCORE,
            color=style.LABEL_TEXT, zorder=12,
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc="white", ec=style.EDGE_GRAY, lw=0.4, alpha=0.92))
    return not all(front[1:])


def main():
    argparse.ArgumentParser().parse_args()
    style.apply()
    data = json.load(open(style.RESULTS / "text_trees.json"))
    trees = data["trees"]
    W = style.TEXT_IN
    pw = (W - OUTLINE - GAP * len(SHOWN)) / len(SHOWN)
    for page in range(math.ceil(len(trees) / TREES_PER_PAGE)):
        index = list(range(page * TREES_PER_PAGE, min(len(trees), (page + 1) * TREES_PER_PAGE)))
        H = TITLE + len(index) * ROW_H + (len(index) - 1) * VGAP + KEY
        fig = plt.figure(figsize=(W, H))
        for col, key in enumerate(SHOWN):
            x0 = OUTLINE + GAP + col * (pw + GAP)
            fig.text((x0 + pw / 2) / W, (H - TITLE + 0.03) / H, MODELS[key], ha="center", va="bottom", fontsize=8)
        any_inward = False
        for row, n in enumerate(index):
            tree = trees[n]
            _, _, colour, marker = branch_style(tree)
            y0 = H - TITLE - (row + 1) * ROW_H - row * VGAP
            ox = fig.add_axes([0, y0 / H, OUTLINE / W, ROW_H / H])
            ox.set_xlim(0, OUTLINE), ox.set_ylim(0, ROW_H)
            ox.axis("off")
            outline(ox, tree, colour, marker, ROW_H)
            for col, key in enumerate(SHOWN):
                res = data["models"][key]["trees"][n]
                bg = np.array(res.get("background_xy") or data["models"][key]["background_xy"])
                x0 = OUTLINE + GAP + col * (pw + GAP)
                ax = fig.add_axes([x0 / W, y0 / H, pw / W, ROW_H / H])
                any_inward |= panel(ax, res, bg, tree, colour, marker, pw, ROW_H)
            print(f"{tree['noun']:7s} in front " + ", ".join(f"{MODELS[k].split()[0]} {data['models'][k]['trees'][n]['in_front']}" for k in SHOWN))

        rend = fig.canvas.get_renderer()

        def put(x, y, text):
            t = fig.text(x / W, y / H, text, ha="left", va="center", fontsize=FS_KEY, color=style.LABEL_TEXT)
            fig.canvas.draw()
            return t.get_window_extent(rend).x1 / fig.dpi

        def line(x, y, **kw):
            fig.add_artist(plt.Line2D([x / W, (x + 0.2) / W], [y / H] * 2, color=style.NODE_STROKE, transform=fig.transFigure, **kw))
            return x + 0.24

        y, x = KEY / 2 - 0.02, OUTLINE + GAP
        x = put(line(x, y, lw=LW, solid_capstyle="round"), y, "child in front") + 0.14
        x = put(line(x, y, lw=LW_DASH, ls=DASH), y, "not in front") + 0.14
        fig.add_artist(plt.Line2D([(x + 0.04) / W], [y / H], marker="o", ms=2.2, ls="", color=BACKGROUND, transform=fig.transFigure))
        x = put(x + 0.10, y, "other texts") + 0.14
        put(x, y, "upward: away from the origin")
        style.save(fig, f"text_tree_plane_all_{page + 1}.pdf")


if __name__ == "__main__":
    main()
