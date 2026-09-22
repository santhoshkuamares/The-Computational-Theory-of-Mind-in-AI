"""Draw the benchmark, review, exposure and architecture figures.

Run analyse.py first so results/ contains the required tables. This script
reads those values and saves figures without changing their scores; the
cognitive layer curves are produced separately by cognitive_analysis.py.
"""

from project_setup import ensure_analysis_inputs

ensure_analysis_inputs()
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from analyse import SYSTEMS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
NAMES = [
    "Base direct",
    "Answer-only direct",
    "Subjectesis direct",
    "Subjectesis no review",
    "Subjectesis review",
]
COLORS = ["#677483", "#1f4e79", "#b16b2c", "#347b78", "#6f587a"]
COLOR = dict(zip(SYSTEMS, COLORS))
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.65,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    """Export a completed figure to PNG, PDF and SVG using the same layout and
    then close it. The repository retains the selected PNGs, while local vector
    exports support resizing or inclusion in the dissertation.
    """
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            OUT / (name + "." + ext), dpi=220, bbox_inches="tight", pad_inches=0.1
        )
    plt.close(fig)


for kind, label, metric, axislabel in [
    (
        "rectom",
        "RecToM",
        "mean_family_accuracy",
        "Mean of belief and desire accuracy (%)",
    ),
    ("opentom", "OpenToM", "accuracy", "Strict question accuracy (%)"),
]:
    d = pd.read_csv(DATA / f"{kind}_systems.csv").set_index("system").loc[SYSTEMS]
    fig, ax = plt.subplots(figsize=(7.1, 3.45), layout="constrained")
    for i, sys in enumerate(SYSTEMS):
        r = d.loc[sys]
        v = r[metric] * 100
        lo = r[metric + "_low"] * 100
        hi = r[metric + "_high"] * 100
        ax.errorbar(
            v,
            4 - i,
            xerr=[[v - lo], [hi - v]],
            fmt="o",
            color=COLOR[sys],
            capsize=3,
            markersize=6,
        )
        ax.text(101, 4 - i, f"{v:.2f}", ha="left", va="center", fontsize=9)
    ax.set_yticks(range(5), NAMES[::-1])
    ax.set_xlim(0, 107)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel(axislabel)
    ax.set_title(
        f"{label}: 621 questions, {(67 if kind == 'rectom' else 27)} source clusters",
        loc="left",
    )
    save(fig, f"{kind}_accuracy")
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharey=True, layout="constrained")
p = pd.read_csv(DATA / "paired_comparisons.csv")
selected = [
    ("subjectesis_direct", "answer_only_direct"),
    ("subjectesis_no_review", "subjectesis_direct"),
    ("subjectesis_review", "subjectesis_no_review"),
]
for ax, kind in zip(axes, ["rectom", "opentom"]):
    metric = "mean_family_accuracy" if kind == "rectom" else "accuracy"
    for i, (a, b) in enumerate(selected):
        r = p[(p.dataset == kind) & (p.a == a) & (p.b == b)].iloc[0]
        v = r[metric + "_difference"] * 100
        lo = r[metric + "_low"] * 100
        hi = r[metric + "_high"] * 100
        ax.errorbar(
            v, 2 - i, xerr=[[v - lo], [hi - v]], color=COLORS[i + 2], fmt="o", capsize=3
        )
        ax.annotate(
            f"{v:+.2f}",
            (v, 2 - i),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlim(-3.5, 4)
    ax.set_ylim(-0.55, 2.6)
    ax.grid(axis="x")
    ax.set_xlabel("Paired difference (percentage points)")
    ax.set_title(
        "RecToM task mean" if kind == "rectom" else "OpenToM strict accuracy",
        loc="left",
    )
axes[0].set_yticks(
    [2, 1, 0],
    [
        "Subjectesis minus answer-only\nDirect inference",
        "State execution minus direct\nSubjectesis adapter",
        "Review minus no review\nShared initial state",
    ],
)
save(fig, "paired_effects")
family_names = {
    "location_cg_fo": "Location, coarse: first order",
    "location_cg_so": "Location, coarse: second order",
    "location_fg_fo": "Location, fine: first order",
    "location_fg_so": "Location, fine: second order",
    "multihop_fo_fullness": "Fullness: first order",
    "multihop_fo_accessibility": "Accessibility: first order",
    "multihop_so_fullness": "Fullness: second order",
    "multihop_so_accessibility": "Accessibility: second order",
    "attitude": "Attitude",
}
d = pd.read_csv(DATA / "opentom_families.csv")
order = list(family_names)
matrix = (
    d.pivot(index="family", columns="system", values="f1").loc[order, SYSTEMS] * 100
)
fig, ax = plt.subplots(figsize=(8.5, 5.3), layout="constrained")
im = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
ax.set_yticks(
    range(9),
    [
        family_names[f] + f" (n={int(d[d.family == f].iloc[0].questions)})"
        for f in order
    ],
)
ax.set_xticks(
    range(5),
    [
        "Base",
        "Answer-only",
        "Subjectesis\ndirect",
        "Subjectesis\nno review",
        "Subjectesis\nreview",
    ],
)
for y in range(9):
    for x in range(5):
        ax.text(
            x,
            y,
            f"{matrix.iloc[y, x]:.1f}",
            ha="center",
            va="center",
            color="white" if matrix.iloc[y, x] > 55 else "#172632",
            fontsize=9,
        )
fig.colorbar(im, ax=ax, shrink=0.8, label="Fixed-class macro-F1 (%)")
ax.set_title("OpenToM performance by question family", loc="left")
save(fig, "opentom_family_f1")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
for ax, kind in zip(axes, ["rectom", "opentom"]):
    m = json.loads((DATA / f"{kind}_process_summary.json").read_text())
    v = [
        m.get(k, 0)
        for k in [
            "wrong_to_right",
            "right_to_wrong",
            "right_to_right",
            "wrong_to_wrong",
        ]
    ]
    bars = ax.barh(range(4), v, color=["#347b78", "#b16b2c", "#a3aab2", "#d0d4d8"])
    ax.set_yticks(
        range(4),
        ["Wrong to right", "Right to wrong", "Right to right", "Wrong to wrong"],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 650)
    ax.set_xlabel("Questions")
    ax.set_title(
        ("RecToM" if kind == "rectom" else "OpenToM") + " review transitions",
        loc="left",
    )
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for b, n in zip(bars, v):
        ax.text(n + 8, b.get_y() + b.get_height() / 2, str(n), va="center", fontsize=9)
save(fig, "review_transitions")
d = pd.read_csv(DATA / "validation_history.csv")
fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
for cond, label, color, marker, best in [
    ("ordinary", "Answer-only", COLORS[1], "o", 1000),
    ("subjectesis", "Subjectesis", COLORS[2], "s", 2500),
]:
    s = d[d.condition == cond]
    ax.plot(
        s.step,
        s.selection_score * 100,
        label=label,
        color=color,
        marker=marker,
        ms=4,
        lw=1.2,
    )
    r = s[s.step == best].iloc[0]
    ax.scatter(
        [best],
        [r.selection_score * 100],
        s=100,
        facecolors="none",
        edgecolors=color,
        lw=1.2,
        zorder=5,
    )
ax.set_xlabel("Optimizer step")
ax.set_ylabel("Validation task-mean accuracy (%)")
ax.set_ylim(70, 100)
ax.grid()
ax.legend(frameon=False)
ax.set_title("Checkpoint selection on 319 questions from 34 dialogues", loc="left")
save(fig, "validation_history")
fig, ax = plt.subplots(figsize=(7.1, 3.2), layout="constrained")
d = pd.read_csv(DATA / "training_audit.csv")
x = np.arange(2)
w = 0.32
for i, (col, label, color) in enumerate(
    [
        ("selected_examples_seen", "Evaluated checkpoint", "#1f4e79"),
        ("completed_examples", "Full run", "#b4bec8"),
    ]
):
    b = ax.bar(x + (i - 0.5) * w, d[col], width=w, color=color, label=label)
    for z in b:
        ax.text(
            z.get_x() + z.get_width() / 2,
            z.get_height() + 450,
            f"{int(z.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
ax.set_xticks(x, ["Answer-only", "Subjectesis"])
ax.set_ylim(0, 36000)
ax.set_ylabel("Training example exposures")
ax.legend(frameon=False, loc="upper left", ncols=2)
ax.grid(axis="y")
ax.set_axisbelow(True)
save(fig, "checkpoint_exposure")
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")
boxes = [
    (0, 1.8, 2, 1.2, "Input evidence\nQuestion\nAnswer format"),
    (
        3.1,
        0.3,
        3,
        1.2,
        "Object-level operations\nQwen with QLoRA adapter\nGenerate or revise state",
    ),
    (3.1, 3.2, 3, 1.3, "Explicit state\nPerspective and claims\nEvidence and unknowns"),
    (7.2, 3.2, 2.7, 1.3, "Controller\nSelect review field\nValidate and merge"),
    (7.2, 0.3, 2.7, 1.2, "Final answer\nSame model\nTask answer format"),
]
for x, y, w, h, txt in boxes:
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="#f2f5f7", edgecolor="#66737f", lw=0.9)
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        txt,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.5,
    )


def arrow(a, b, label=None, pos=None):
    """Draw a directed connection between two coordinates on the current
    architecture axes. Optionally place a short label at a supplied position so
    the diagram explains the flow between the implemented components.
    """
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=11, lw=1, color="#334f65"
        )
    )
    if label:
        ax.text(*pos, label, ha="center", fontsize=9, color="#334f65")


arrow((2, 2.2), (3.1, 1.1))
arrow((4.1, 1.5), (4.1, 3.2), "Monitoring", (3.5, 2.3))
arrow((6.1, 3.85), (7.2, 3.85))
arrow((8, 3.2), (5.5, 1.5), "Review instruction", (7.1, 2.2))
arrow((6.1, 0.9), (7.2, 0.9))
ax.text(
    5,
    4.9,
    "Functional monitoring and control with bounded review",
    ha="center",
    fontsize=12,
)
ax.text(
    5,
    -0.2,
    "The controller is hand-written. The same neural model performs state generation, review and answering.",
    ha="center",
    fontsize=9,
)
save(fig, "subjectesis_architecture")
d = pd.read_csv(DATA / "opentom_systems.csv").set_index("system").loc[SYSTEMS]
fig, ax = plt.subplots(figsize=(7.1, 3.45), layout="constrained")
for i, sys in enumerate(SYSTEMS):
    r = d.loc[sys]
    v = r.mean_family_f1 * 100
    lo = r.mean_family_f1_low * 100
    hi = r.mean_family_f1_high * 100
    ax.errorbar(
        v,
        4 - i,
        xerr=[[v - lo], [hi - v]],
        fmt="o",
        color=COLOR[sys],
        capsize=3,
        markersize=6,
    )
    ax.text(101, 4 - i, f"{v:.2f}", va="center", fontsize=9)
ax.set_yticks(range(5), NAMES[::-1])
ax.set_xlim(0, 107)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.grid(axis="x")
ax.set_xlabel("Mean family fixed-class macro-F1 (%)")
ax.set_title("OpenToM: study-defined mean over nine families", loc="left")
save(fig, "opentom_macro_f1")
print("Exported nine figure sets in PNG, PDF and SVG.")
