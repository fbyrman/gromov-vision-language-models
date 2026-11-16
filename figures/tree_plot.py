"""Helpers shared by the text-tree figures: colours, the origin mark, geodesics of the Poincare disk."""
import numpy as np

import style

BACKGROUND = "#C9CDD2"
MODELS = {"gromovclip": "GromovCLIP (ours)", "hycoclip": "HyCoCLIP", "argent": "ARGENT", "phyclip": "PHyCLIP"}
# One hue per tree, a darker and a lighter step of it for its two branches; dark enough for white digits.
# The methods' blue and orange are left out. The root takes the darker step.
PALETTE = [("#332288", "#7B70C4"), ("#882255", "#CC6677"), ("#117733", "#5FA86B"), ("#7A5C3E", "#B39468"),
           ("#8A3A1E", "#C98A6F"), ("#1B6B7A", "#6FB3BF"), ("#4D4D8F", "#9A9AD0"), ("#6B6B1F", "#B5B56A")]
# Branch encoding of the tree figures: the first branch red circles, the second blue triangles, light at level 2 and
# strong at level 4 (ColorBrewer RdBu); the root is grey.
RAMP = [["#F4A582", "#D6604D", "#B2182B"], ["#92C5DE", "#4393C3", "#2166AC"]]
MARKER, MS = ["o", "^"], [3.4, 4.0]
ROOT, ROOT_MS = "#5A5F66", 3.8
ZOOM_RED = "#D1193E"     # a magnified panel's frame and its box in the whole disk: the same region at two scales
DASH = (0, (2.2, 1.1))   # in units of the line width
LW, LW_DASH = 1.0, 1.35  # dashed steps a little heavier, so short ones stay visible
FS_BADGE = 3.9
BADGE = dict(boxstyle="circle,pad=0.17", ec="white", lw=0.35)


def origin_mark(ax, ms=3.2, fontsize=7, offset=(-2.5, -2.5), zorder=2):
    """The origin as in the other Poincare disks of the paper: a point in the root fill, labelled with a bold zero."""
    ax.plot(0, 0, "o", ms=ms, mfc=style.ROOT_FILL, mec=style.NODE_STROKE, mew=0.14 * ms, zorder=zorder, clip_on=True)
    ax.annotate(r"$\mathbf{0}$", (0, 0), xytext=offset, textcoords="offset points", ha="right", va="top" if offset[1] < 0 else "bottom", fontsize=fontsize,
                color=style.LABEL_TEXT, zorder=zorder, annotation_clip=True)


def mobius(x, y):
    xy, x2, y2 = (x * y).sum(-1, keepdims=True), (x * x).sum(-1, keepdims=True), (y * y).sum(-1, keepdims=True)
    return ((1 + 2 * xy + y2) * x + (1 - x2) * y) / (1 + 2 * xy + x2 * y2)


def geodesic(a, b, n=48):
    """Geodesic of the Poincare disk from a to b."""
    v = mobius(-a[None], b[None])[0]
    r = np.linalg.norm(v)
    t = np.linspace(0, 1, n)[:, None]
    return mobius(np.repeat(a[None], n, 0), np.tanh(t * np.arctanh(r)) * v / r)


def window(pts, width, height):
    """Axis limits around the points with the aspect of the panel."""
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.035 * (hi - lo)
    xl, yl = [lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]
    if (xl[1] - xl[0]) / (yl[1] - yl[0]) > width / height:
        need, c = (xl[1] - xl[0]) * height / width, sum(yl) / 2
        yl = [c - need / 2, c + need / 2]
    else:
        need, c = (yl[1] - yl[0]) * width / height, sum(xl) / 2
        xl = [c - need / 2, c + need / 2]
    return xl, yl


def node_colours(tree, pair):
    """Colour of every node of a tree: the step of its branch; the root takes the darker step."""
    nodes, out, branch = tree["nodes"], [], -1
    for node in nodes:
        branch += node["level"] == 2
        out.append(pair[max(branch, 0)])
    return out


def branch_style(tree):
    """Colour, marker and marker size of every node of a tree: the branch gives the hue and marker, the level the strength."""
    level = np.array([n["level"] for n in tree["nodes"]])
    branch = np.maximum(np.cumsum(level == 2) - 1, 0)
    colour = [ROOT if level[j] == 1 else RAMP[branch[j]][level[j] - 2] for j in range(len(level))]
    marker = [("o", ROOT_MS) if level[j] == 1 else (MARKER[branch[j]], MS[branch[j]]) for j in range(len(level))]
    return level, branch, colour, marker
