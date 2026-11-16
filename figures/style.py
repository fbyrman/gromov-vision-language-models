"""Shared figure style: sizes, colours and fonts of the paper's figures."""
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "outputs" / "figures"

COLUMN_IN = 3.25
TEXT_IN = 6.875

EDGE_GRAY = "#C7CBD0"
NODE_STROKE = "#6B7077"
ROOT_FILL = "#AFC6DE"
MARK_FILL = "#E3B7B0"
LABEL_TEXT = "#2A2E33"
GROMOV = "#0072B2"   # GromovCLIP
REST = "#E5651A"     # baselines

# One sequential ramp per loss family, in the colour of its methods: GROMOV for the Gromov loss, REST for the
# entailment and adaptive losses of the baselines. Same tint/shade steps, so lightness means the same in both.
GROMOV_STOPS = ["#EDF5FA", "#80B9D9", GROMOV, "#00375A"]
REST_STOPS = ["#FDF5EF", "#F3B48A", REST, "#6E2A05"]
GROMOV_CMAP = LinearSegmentedColormap.from_list("loss_gromov", GROMOV_STOPS)
REST_CMAP = LinearSegmentedColormap.from_list("loss_rest", REST_STOPS)
LOSS_CMAP = GROMOV_CMAP
# Gradient magnitude: blue where the gradient is small, orange where large.
GRAD_CMAP = LinearSegmentedColormap.from_list("grad", ["#7FB2DA", "#F4F1EC", "#F2A866", "#E5651A"])


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": NODE_STROKE,
        "axes.linewidth": 0.6,
        "hatch.color": NODE_STROKE,
        "hatch.linewidth": 0.4,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def save(fig, name):
    """Write a figure to outputs/figures/ as a vector PDF."""
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path.relative_to(REPO)}")
