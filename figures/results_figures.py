"""Paper C: the results figures in the v3 style. Version 3.1, 12 October 2026 (from PaperC_result_figures_v3_0.py).
CPU only, a few seconds. No data files are needed: every number below is the exact value from the result files
(v3.0 read some of them off the earlier figures), and --check compares them with the files when these are present.

All figure text: Times New Roman (Liberation Serif, metric-compatible, as fallback), regular, black, 8 pt at printed
size. Figures are drawn at their printed width and saved at 600 dpi.
  Figure 5   paired outcomes, ours minus baseline (Table 3)            PaperC_Fig05_paired_v3_1.png    6.3 in wide
  Figure 6   (a) per-class IoU change, (b) change against overlap      PaperC_Fig06_perclass_v3_1.png  6.3 in
  Figure 9   per-class IoU at full size and at 512 x 512, same predictions  PaperC_Fig09_scoring_grid_v3_1.png  4.3 in
  Figure 10  external test sets without retraining (S2DS, DeepCrack)   PaperC_Fig10_external_v3_1.png  6.3 in

Data sources (exact)
  Table 3 rows and the per-class intervals: paired bootstrap over the 975 validation images, 2,000 resamples
      (bootstrap.py; the same numbers as Table 3 and Supplementary S4).
  Per-class IoU at full size and 512 x 512: results_figpack_v1_1/per_class_iou_v1_1.csv (run B3-ovl-L0.1-s0, TTA).
  Overlap share: share of the class's training pixels that carry another class (Section 3.2).
  S2DS and DeepCrack: results_xdata_v1_0/*.json (test_all_L) and zero_shot_bootstrap_v1_0.txt.
Usage
  python results_figures.py                 figures into ./figs_v3_1
  python results_figures.py --out DIR       figures into DIR
  python results_figures.py --check         also compare the numbers with the result files next to it
"""
import argparse, csv, json, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.weight": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
    "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.edgecolor": "black", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "mathtext.fontset": "stix",
    "savefig.dpi": 600, "savefig.facecolor": "white", "pdf.fonttype": 42,
})
BLUE = "#2166AC"     # proposed model / with TTA
ORANGE = "#E66101"   # baseline / thin and small classes / without TTA
GREY = "#8C8C8C"
LIGHT = "#EEF3F9"
RED = "#C0392B"
GRID = "#DDDDDD"

# ------------------------------------------------------------------------------------------------ exact data
# Table 3: outcome, ours (TTA, %), baseline (TTA, %), (d, lo, hi) with TTA, (d, lo, hi) without TTA, as fractions
PAIRED = [
    ("Exact label-set match $M$ (primary)", 40.4, 37.1, (.0329, .0061, .0603), (.0385, .0091, .0714)),
    ("Pair IoU, ExposedRebars & Rust", 44.8, 42.8, (.0204, .0016, .0421), (.0203, -.0038, .0499)),
    ("Pair IoU, ExposedRebars & Spalling", 43.8, 42.9, (.0090, -.0115, .0309), (.0060, -.0171, .0353)),
    ("Pair IoU, Spalling & Rust", 50.4, 50.2, (.0015, -.0167, .0209), (.0008, -.0151, .0164)),
    ("mIoU, full size", 50.0, 49.7, (.0036, -.0020, .0094), (.0048, -.0005, .0103)),
    ("mIoU, 512 $\\times$ 512", 49.3, 49.6, (-.0022, -.0063, .0020), (-.0024, -.0067, .0020)),
    ("Coverage recall, Cavity", 47.3, 45.1, (.0217, -.0009, .0461), (.0330, .0130, .0518)),
    ("Coverage recall, ExposedRebars", 77.3, 74.5, (.0283, -.0163, .0738), (.0364, .0000, .0750)),
]
GROUPS = [(0, 4, "Label sets and intersections"), (4, 6, "Per-class accuracy"), (6, 8, "Small-damage detection")]
# per-class IoU, ours minus baseline, full size, TTA: (d, lo, hi), fractions
PER_CLASS = {
    "Crack": (.0011, -.0090, .0132), "ACrack": (.0248, -.0052, .0550), "Wetspot": (.0272, .0048, .0562),
    "Efflorescence": (.0178, .0030, .0339), "Rust": (-.0046, -.0154, .0067), "Rockpocket": (-.0134, -.0468, .0177),
    "Hollowareas": (.0047, -.0222, .0270), "Cavity": (.0065, -.0149, .0259), "Spalling": (.0099, -.0177, .0400),
    "Graffiti": (-.0038, -.0113, .0022), "Weathering": (.0223, -.0107, .0675), "Restformwork": (-.0276, -.0793, .0097),
    "ExposedRebars": (.0190, .0005, .0403), "Bearing": (-.0031, -.0218, .0149), "EJoint": (.0084, -.0295, .0486),
    "Drainage": (-.0082, -.0471, .0130), "PEquipment": (-.0040, -.0164, .0061), "JTape": (-.0008, -.0212, .0248),
    "WConccor": (-.0080, -.0425, .0303)}
THIN = {"Crack", "Cavity", "ExposedRebars", "Rust"}          # thin and small classes (T)
OVERLAP_SHARE = {  # share of the class's training pixels that carry another class (%)
    "Crack": 43.3, "ACrack": 63.2, "Wetspot": 64.1, "Efflorescence": 38.0, "Rust": 63.4, "Rockpocket": 30.2,
    "Hollowareas": 40.4, "Cavity": 32.7, "Spalling": 39.2, "Graffiti": 12.3, "Weathering": 26.7, "Restformwork": 28.6,
    "ExposedRebars": 99.3, "Bearing": 9.8, "EJoint": 30.1, "Drainage": 35.1, "PEquipment": 7.6, "JTape": 39.4,
    "WConccor": 25.2}
# IoU of our model (TTA) with the labels drawn at full size and at 512 x 512: per_class_iou_v1_1.csv
GRID_IOU = {
    "Crack": (.3648, .3507), "ACrack": (.5818, .5331), "Wetspot": (.3737, .3455), "Efflorescence": (.5479, .4352),
    "Rust": (.5844, .5210), "Rockpocket": (.1655, .1818), "Hollowareas": (.6922, .6505), "Cavity": (.2124, .2304),
    "Spalling": (.6085, .4933), "Graffiti": (.5526, .6586), "Weathering": (.4053, .4438), "Restformwork": (.3709, .4393),
    "ExposedRebars": (.4518, .5313), "Bearing": (.7733, .7461), "EJoint": (.6837, .6960), "Drainage": (.7273, .7497),
    "PEquipment": (.7995, .7776), "JTape": (.4266, .4752), "WConccor": (.1860, .1133)}
# S2DS test, input long side 1024 (both models chose 1024); crack: line-based IoU with 4-pixel tolerance
S2DS = {"classes": ["crack", "spalling", "corrosion", "efflorescence"],
        "baseline": [.5707, .7944, .5833, .6798], "ours": [.5919, .7873, .5987, .6865]}
# DeepCrack test, crack IoU at input long side 512, 768, 1024; chosen size on labelled non-test images
DEEPCRACK = {"sizes": [512, 768, 1024], "baseline": [.2044, .2182, .2393], "ours": [.2189, .2322, .2484],
             "chosen": {"baseline": 768, "ours": 1024}, "same_size_1024": (.0091, .0014, .0177)}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def pts(v):
    """fraction to signed points, rounded half away from zero as in the tables, with a true minus sign; 0.0 unsigned."""
    q = (Decimal(repr(round(float(v), 6))) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return "0.0" if q == 0 else f"{q:+}".replace("-", "−")


def tick(v):
    """integer tick label with a sign and a true minus sign."""
    return "0" if v == 0 else f"{v:+d}".replace("-", "−")


def resolved(lo, hi):
    return lo > 0 or hi < 0


# ---------------------------------------------------------------------------------------------- Figure 5
def fig_paired(out):
    rows = PAIRED
    fig = plt.figure(figsize=(6.3, 3.25))
    ax = fig.add_axes([0.285, 0.17, 0.255, 0.70])
    ypos, y = [], 0.0
    for a, b, _ in GROUPS:
        for _ in range(a, b):
            ypos.append(y); y += 1.0
        y += 0.55
    ypos = np.array(ypos)
    ax.axhspan(ypos[0] - 0.5, ypos[0] + 0.5, color=LIGHT, zorder=0, lw=0)
    ax.axvline(0, color="black", lw=0.6, zorder=1)
    for x in (-2, 2, 4, 6, 8):
        ax.axvline(x, color=GRID, lw=0.4, zorder=0)
    for i, (_, _, _, t, nt) in enumerate(rows):
        for (d, lo, hi), off, colr, mk in ((t, -0.17, BLUE, "o"), (nt, 0.17, ORANGE, "s")):
            ax.plot([100 * lo, 100 * hi], [ypos[i] + off] * 2, color=colr, lw=1.1, solid_capstyle="butt", zorder=2)
            ax.plot(100 * d, ypos[i] + off, marker=mk, ms=4.2, mew=0.9, mec=colr,
                    mfc=colr if resolved(lo, hi) else "white", zorder=3)
    ax.set_ylim(ypos[-1] + 0.7, ypos[0] - 0.7)
    ax.set_xlim(-2.5, 8.0)
    ax.set_xticks([-2, 0, 2, 4, 6, 8])
    ax.set_xticklabels([tick(v) for v in (-2, 0, 2, 4, 6, 8)])
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Proposed minus baseline (points)", labelpad=2)
    style(ax)
    ax.spines["left"].set_visible(False)
    for a, _, g in GROUPS:                       # group headers above each group, at the left figure edge
        yh = ax.transData.transform((0, ypos[a] - 0.62))[1] / fig.bbox.height
        fig.text(0.005, yh, g + ":", va="center", ha="left", style="italic")
    tx = ax.transAxes                            # numeric columns to the right, in axes fractions
    cols_x = [1.06, 1.50, 2.04]
    heads = ["Proposed /\nbaseline (TTA)", "Difference, TTA\n[95% interval]", "Difference, no TTA\n[95% interval]"]
    for cx, h in zip(cols_x, heads):
        ax.text(cx, 1.005, h, transform=tx, ha="left", va="bottom", linespacing=1.0)
    for i, (_, p, b, t, nt) in enumerate(rows):
        yy = tx.inverted().transform(ax.transData.transform((0, ypos[i])))[1]
        ax.text(cols_x[0], yy, f"{p:.1f} / {b:.1f}", transform=tx, va="center", ha="left")
        for cx, (d, lo, hi) in ((cols_x[1], t), (cols_x[2], nt)):
            ax.text(cx, yy, f"{pts(d)} [{pts(lo)}, {pts(hi)}]", transform=tx, va="center", ha="left")
    handles = [Line2D([], [], color=BLUE, marker="o", ms=4.2, lw=1.1, label="With TTA"),
               Line2D([], [], color=ORANGE, marker="s", ms=4.2, lw=1.1, label="Without TTA"),
               Line2D([], [], color="black", marker="o", ms=4.2, mfc="white", lw=0,
                      label="Open marker: interval includes zero")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=3,
               handlelength=1.6, columnspacing=1.6, borderaxespad=0.2)
    fig.savefig(os.path.join(out, "PaperC_Fig05_paired_v3_1.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- Figure 6
def fig_perclass(out):
    order = sorted(PER_CLASS, key=lambda c: -PER_CLASS[c][0])
    fig = plt.figure(figsize=(6.3, 3.3))
    ax = fig.add_axes([0.125, 0.125, 0.30, 0.80])
    ax.axvline(0, color="black", lw=0.6)
    for x in (-6, -4, -2, 2, 4, 6):
        ax.axvline(x, color=GRID, lw=0.4, zorder=0)
    for i, c in enumerate(order):
        d, lo, hi = PER_CLASS[c]
        colr = ORANGE if c in THIN else BLUE
        ax.plot([100 * lo, 100 * hi], [i, i], color=colr, lw=1.1, solid_capstyle="butt")
        ax.plot(100 * d, i, "o", ms=4.0, mew=0.9, mec=colr, mfc=colr if resolved(lo, hi) else "white")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(-8.5, 7.3)
    ax.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6])
    ax.set_xticklabels([tick(v) for v in (-8, -6, -4, -2, 0, 2, 4, 6)])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel("IoU change, proposed minus baseline (points)")
    fig.text(0.005, 0.975, "(a) Per-class IoU change with 95% interval", va="top", ha="left")
    style(ax)
    ax.spines["left"].set_visible(False)

    bx = fig.add_axes([0.565, 0.125, 0.42, 0.80])
    xs = np.array([OVERLAP_SHARE[c] for c in order])
    ys = np.array([100 * PER_CLASS[c][0] for c in order])
    rho, p = spearmanr(xs, ys)
    hi_mean, n_hi = ys[xs >= 35].mean(), int((xs >= 35).sum())
    for v in (-2, -1, 1, 2):
        bx.axhline(v, color=GRID, lw=0.4, zorder=0)
    for v in (20, 40, 60, 80):
        bx.axvline(v, color=GRID, lw=0.4, zorder=0)
    bx.axhline(0, color="black", lw=0.6)
    bx.axvspan(35, 102, color=LIGHT, lw=0, zorder=0)
    for c, x_, y_ in zip(order, xs, ys):
        colr = ORANGE if c in THIN else BLUE
        bx.plot(x_, y_, "o", ms=4.6, mew=0.9, mec=colr, mfc=colr if resolved(*PER_CLASS[c][1:]) else "white", zorder=3)
    labels = {"Wetspot": (4, 1), "ACrack": (-36, -3), "Weathering": (-46, -3), "ExposedRebars": (-58, 6),
              "Efflorescence": (4, 2), "Spalling": (4, 2), "Rust": (4, -8), "Rockpocket": (4, -7),
              "Restformwork": (-55, -3), "Crack": (4, 2), "Cavity": (-30, -10), "EJoint": (-30, 2)}
    for c, x_, y_ in zip(order, xs, ys):
        if c in labels:
            bx.annotate(c, (x_, y_), xytext=labels[c], textcoords="offset points")
    small = sorted([c for c in order if OVERLAP_SHARE[c] < 15], key=len)      # Bearing, Graffiti, PEquipment
    bx.text(0.8, -0.58, ",\n".join(small), va="top", ha="left", linespacing=1.0)
    bx.text(62, -0.95, f"Shaded: at least 35%\noverlapping pixels;\nmean change {pts(hi_mean / 100)}",
            va="top", linespacing=1.0)
    bx.text(1.5, 3.17, f"Spearman $\\rho$ = {rho:.2f}\n(p = {p:.2f}), {len(order)} classes", va="top", linespacing=1.0)
    bx.set_xlim(0, 102)
    bx.set_ylim(-3.2, 3.2)
    bx.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    bx.set_yticklabels([tick(v) for v in (-3, -2, -1, 0, 1, 2, 3)])
    bx.set_xlabel("Share of the class's training pixels that carry another class (%)")
    bx.set_ylabel("IoU change (points)")
    fig.text(0.50, 0.975, "(b) IoU change against overlap share", va="top", ha="left")
    style(bx)
    handles = [Line2D([], [], color=BLUE, marker="o", lw=0, ms=4.4, label="Other classes"),
               Line2D([], [], color=ORANGE, marker="o", lw=0, ms=4.4, label="Thin and small classes (T)"),
               Line2D([], [], color="black", marker="o", mfc="white", lw=0, ms=4.4, label="Open: interval includes zero")]
    bx.legend(handles=handles, loc="lower right", handletextpad=0.3, borderaxespad=0.1)
    fig.savefig(os.path.join(out, "PaperC_Fig06_perclass_v3_1.png"))
    plt.close(fig)
    return rho, p, hi_mean, n_hi


# ---------------------------------------------------------------------------------------------- Figure 9
def fig_grid(out):
    """the five largest falls and the five largest rises from full-size to 512 x 512 scoring, same predictions."""
    order = sorted(GRID_IOU, key=lambda c: GRID_IOU[c][1] - GRID_IOU[c][0])
    sel = order[:5] + order[-5:]
    rows = [(c, GRID_IOU[c][0], GRID_IOU[c][1] - GRID_IOU[c][0]) for c in sel]
    fig, ax = plt.subplots(figsize=(4.3, 2.95))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.27, top=0.98)
    for i, (c, f, d) in enumerate(rows):
        g = f + d
        colr = RED if d < 0 else BLUE
        ax.plot([f, g], [i, i], color=colr, lw=1.6, solid_capstyle="butt", zorder=2)
        ax.plot(f, i, "o", ms=4.4, mfc="white", mec="black", mew=0.9, zorder=3)
        ax.plot(g, i, "s", ms=4.0, mfc=colr, mec=colr, zorder=3)
        ax.text(max(f, g) + 0.012, i, pts(d), va="center", ha="left")
    for x in np.arange(0.1, 0.81, 0.1):
        ax.axvline(x, color=GRID, lw=0.4, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0.05, 0.80)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Validation IoU of the proposed model (same predictions, TTA)")
    style(ax)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", mfc="white", mec="black", lw=0, ms=4.4, label="Labels drawn at full size"),
               Line2D([], [], marker="s", color=GREY, lw=0, ms=4.0, label="Labels drawn at 512 $\\times$ 512"),
               Patch(color=RED, label="Lower at 512 $\\times$ 512"), Patch(color=BLUE, label="Higher at 512 $\\times$ 512")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=2, handlelength=1.2,
               handletextpad=0.4, labelspacing=0.3, columnspacing=1.5)
    fig.savefig(os.path.join(out, "PaperC_Fig09_scoring_grid_v3_1.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- Figure 10
def fig_external(out):
    fig = plt.figure(figsize=(6.3, 2.1))
    ax = fig.add_axes([0.075, 0.2, 0.40, 0.72])
    cls = ["Crack\n(line-based IoU)", "Spalling", "Corrosion", "Efflorescence"]
    base, prop = S2DS["baseline"], S2DS["ours"]
    x, w = np.arange(4), 0.36
    ax.bar(x - w / 2, base, w, color=ORANGE, label="Baseline")
    ax.bar(x + w / 2, prop, w, color=BLUE, label="Proposed")
    for xi, b, p in zip(x, base, prop):
        ax.text(xi - w / 2, b + 0.012, f"{b:.3f}", ha="center", va="bottom", rotation=90)
        ax.text(xi + w / 2, p + 0.012, f"{p:.3f}", ha="center", va="bottom", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(cls, linespacing=1.0)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("IoU, S2DS test (93 patches)")
    ax.set_title("(a) S2DS, input long side 1024", loc="left")
    ax.legend(loc="upper right", ncol=2, handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(1.0, 1.04))
    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.axhline(v, color=GRID, lw=0.4, zorder=0)
    style(ax)

    bx = fig.add_axes([0.60, 0.2, 0.38, 0.72])
    sizes, pb, pp = DEEPCRACK["sizes"], DEEPCRACK["baseline"], DEEPCRACK["ours"]
    bx.plot(sizes, pb, "-s", color=ORANGE, ms=4.0, lw=1.1, label="Baseline")
    bx.plot(sizes, pp, "-o", color=BLUE, ms=4.2, lw=1.1, label="Proposed")
    for key, vals in (("baseline", pb), ("ours", pp)):
        L = DEEPCRACK["chosen"][key]
        bx.plot(L, vals[sizes.index(L)], "o", ms=9, mfc="none", mec="black", mew=0.7)
    d, lo, hi = DEEPCRACK["same_size_1024"]
    f3 = lambda v: f"{v:+.3f}".replace("-", "−")
    bx.text(1050, 0.2005, f"Same input 1024,\nproposed minus baseline:\n{f3(d)} [{f3(lo)}, {f3(hi)}]",
            ha="right", va="bottom", linespacing=1.0)
    bx.set_xticks(sizes)
    bx.set_xlim(470, 1066)
    bx.set_ylim(0.198, 0.262)
    bx.set_xlabel("Input long side (pixels)")
    bx.set_ylabel("Crack IoU, DeepCrack test (237)")
    bx.set_title("(b) DeepCrack, crack IoU against input size", loc="left")
    for v in (0.21, 0.22, 0.23, 0.24, 0.25):
        bx.axhline(v, color=GRID, lw=0.4, zorder=0)
    h = [Line2D([], [], color=ORANGE, marker="s", ms=4, lw=1.1, label="Baseline"),
         Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.1, label="Proposed"),
         Line2D([], [], marker="o", ms=8, mfc="none", mec="black", lw=0, label="Size chosen on non-test images")]
    bx.legend(handles=h, loc="upper left", handletextpad=0.4, labelspacing=0.3, borderaxespad=0.1)
    style(bx)
    fig.savefig(os.path.join(out, "PaperC_Fig10_external_v3_1.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- check
def check():
    """compare the numbers above with the result files, where these are present next to the script."""
    n = 0
    f = os.path.join(HERE, "..", "results", "per_class_iou_v1_1.csv")
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            if r["run"] == "B3-ovl-L0.1-s0" and r["TTA"] == "TTA":
                assert GRID_IOU[r["class"]] == (float(r["IoU_full"]), float(r["IoU_512"])), r["class"]; n += 1
    X = os.path.join(HERE, "..", "results")
    for key, run in (("baseline", "B3-L-s0"), ("ours", "B3-ovl-L0.1-s0")):
        f = os.path.join(X, f"s2ds_{run}.json")
        if os.path.exists(f):
            t = json.load(open(f))["test_all_L"]["1024"]
            vals = [t["crack_ltIoU"]["4"]] + [t["IoU"][c] for c in S2DS["classes"][1:]]
            assert np.allclose(vals, S2DS[key]), (key, vals); n += 4
        f = os.path.join(X, f"deepcrack_{run}.json")
        if os.path.exists(f):
            j = json.load(open(f))
            vals = [j["test_all_L"][str(L)]["IoU"]["crack"] for L in DEEPCRACK["sizes"]]
            assert np.allclose(vals, DEEPCRACK[key]), (key, vals)
            assert j["chosen_L"] == DEEPCRACK["chosen"][key], (key, j["chosen_L"]); n += 4
    print(f"check: {n} values agree with the result files" if n else "check: no result files found next to the script")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figs_v3_1"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.check:
        check()
    fig_paired(a.out)
    rho, p, m, k = fig_perclass(a.out)
    print(f"Figure 6: Spearman rho {rho:.3f} (p = {p:.3f}); {k} classes with at least 35% overlap, mean change {m:+.2f}")
    fig_grid(a.out)
    fig_external(a.out)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
