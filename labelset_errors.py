"""
Paper C - label-set error analysis (the pending analysis of Section 5).

What it measures, on the validation pixels that carry two or more labels (the set Omega):
  exact    predicted set equals the labelled set
  missing  some labelled classes predicted, some left out, nothing extra
  extra    every labelled class predicted plus at least one class that is not labelled
  mixed    some labelled classes missing and some extra classes added
  none     none of the labelled classes predicted
It also reports
  - M for pixels with exactly two labels and with three or more labels,
  - false overlap: pixels with fewer than two labels where two or more classes are predicted,
  - per-image M of both models (how the gain is spread over images),
  - optionally, the baseline-to-proposed transition matrix of the five categories,
with paired bootstrap intervals over images (2,000 resamples, same images for both models).

HOW TO USE (Colab)
------------------
Your evaluation loop already draws full-size labels (one binary layer per class) and thresholds the
full-size probabilities at 0.5 to compute M. Add three lines to that loop:

    from PaperC_labelset_error_analysis_v1_0 import LabelSetRecorder
    rec = LabelSetRecorder()                                  # before the loop
    ...
    # inside the loop, for every validation image:
    #   labels    : bool array (19, H, W), full-size labels
    #   pred_base : bool array (19, H, W), baseline prediction (probability > 0.5, TTA)
    #   pred_prop : bool array (19, H, W), proposed prediction (probability > 0.5, TTA)
    rec.add(image_name, labels, pred_base, pred_prop)
    ...
    rec.save("labelset_per_image.csv")                       # after the loop

If you can only run one model per loop, call rec.add(image_name, labels, pred_base, None) in the first
run and rec.add(image_name, labels, None, pred_prop) in a second run with a second recorder, save both
CSVs and pass both to analyse(); the transition matrix is then skipped.

Then run the analysis (prints the numbers and writes a summary CSV and the figure):

    !python labelset_errors.py labelset_per_image.csv
    # or, with two separate runs:
    !python labelset_errors.py base.csv prop.csv

Send back: labelset_summary_v1_0.csv (and the printed output). The figure
PaperC_Fig_labelset_v1_0.png is drawn in the manuscript style (Times New Roman, 8 pt, regular, black
text, 600 dpi).

Memory: one full-size image (4032 x 3024) needs about 0.3 GB; images are processed one at a time.
"""
import sys
import csv
import numpy as np

CATS = ["exact", "missing", "extra", "mixed", "none"]
MIN_OMEGA_PX = 500       # images with fewer overlap pixels are left out of the per-image comparison
N_BOOT = 2000
SEED = 2026


# ----------------------------------------------------------------------------- per-image counting
def _codes(masks):
    """(C, H, W) bool -> (H, W) uint32 bit code of the label set, and its size."""
    masks = np.asarray(masks, dtype=bool)
    code = np.zeros(masks.shape[1:], dtype=np.uint32)
    for k in range(masks.shape[0]):
        code |= masks[k].astype(np.uint32) << np.uint32(k)
    size = masks.sum(0, dtype=np.uint8)
    return code, size


def _categories(L, P):
    """Category index 0..4 (CATS order) for every pixel of the given 1-D code arrays."""
    inter = P & L
    exact = P == L
    none = inter == 0
    no_extra = (P & ~L) == 0
    no_missing = (L & ~P) == 0
    cat = np.full(L.shape, 3, dtype=np.uint8)              # mixed by default
    cat[no_extra & ~exact] = 1                              # missing
    cat[no_missing & ~exact] = 2                            # extra
    cat[none] = 4                                           # none (overrides missing when P is empty)
    cat[exact] = 0
    return cat


def _one_model(L, Lsize, P, Psize):
    omega = Lsize >= 2
    l, p = L[omega], P[omega]
    cat = _categories(l, p)
    s = Lsize[omega]
    out = {"n_omega": int(omega.sum()), "n_omega2": int((s == 2).sum()), "n_omega3": int((s >= 3).sum())}
    for i, c in enumerate(CATS):
        out[c] = int((cat == i).sum())
    out["exact2"] = int(((cat == 0) & (s == 2)).sum())
    out["exact3"] = int(((cat == 0) & (s >= 3)).sum())
    out["pred_overlap"] = int((Psize >= 2).sum())
    out["false_overlap"] = int(((Lsize < 2) & (Psize >= 2)).sum())
    return out, cat


class LabelSetRecorder:
    def __init__(self):
        self.rows = []

    def add(self, name, labels, pred_base=None, pred_prop=None):
        L, Ls = _codes(labels)
        row = {"image": name}
        cats = {}
        for tag, pred in (("base", pred_base), ("prop", pred_prop)):
            if pred is None:
                continue
            P, Ps = _codes(pred)
            res, cat = _one_model(L, Ls, P, Ps)
            cats[tag] = cat
            for k, v in res.items():
                row[f"{tag}_{k}"] = v
        if len(cats) == 2:                                   # transition matrix baseline -> proposed
            t = cats["base"].astype(np.int64) * 5 + cats["prop"]
            counts = np.bincount(t, minlength=25)
            for i in range(5):
                for j in range(5):
                    row[f"tr_{CATS[i]}_{CATS[j]}"] = int(counts[i * 5 + j])
        self.rows.append(row)

    def save(self, path):
        keys = sorted({k for r in self.rows for k in r}, key=lambda k: (k != "image", k))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        print(f"saved {len(self.rows)} images to {path}")


# ----------------------------------------------------------------------------- analysis
def _load(paths):
    rows = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                d = rows.setdefault(r["image"], {"image": r["image"]})
                for k, v in r.items():
                    if k != "image" and v != "":
                        d[k] = float(v)
    return list(rows.values())


def _stats(A, tag):
    """Outcome dictionary from a column-sum vector dict A for one model."""
    g = lambda k: A[f"{tag}_{k}"]
    o = {}
    o["M"] = g("exact") / g("n_omega")
    o["M_2labels"] = g("exact2") / g("n_omega2")
    o["M_3plus"] = g("exact3") / g("n_omega3") if g("n_omega3") > 0 else np.nan
    for c in CATS[1:]:
        o[f"share_{c}"] = g(c) / g("n_omega")
    o["false_overlap_share_of_predicted_overlap"] = g("false_overlap") / max(g("pred_overlap"), 1)
    return o


def analyse(paths, fig_path="PaperC_Fig_labelset_v1_0.png", out_csv="labelset_summary_v1_0.csv"):
    rows = _load(paths)
    keys = sorted({k for r in rows for k in r if k != "image"})
    X = np.array([[r.get(k, 0.0) for k in keys] for r in rows])
    idx = {k: i for i, k in enumerate(keys)}
    total = lambda sel: {k: X[sel, i].sum() for k, i in idx.items()}
    n = len(rows)
    allsel = np.arange(n)
    T = total(allsel)
    base, prop = _stats(T, "base"), _stats(T, "prop")
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in base}
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        Tb = total(s)
        b, p = _stats(Tb, "base"), _stats(Tb, "prop")
        for k in base:
            boot[k].append(p[k] - b[k])
    lines = []
    print(f"\n{n} validation images, {int(T['base_n_omega'])} overlap pixels "
          f"({int(T['base_n_omega2'])} with two labels, {int(T['base_n_omega3'])} with three or more)\n")
    print(f"{'outcome':45s}{'baseline':>10s}{'proposed':>10s}{'diff (pts)':>12s}   95% interval")
    for k in base:
        d = 100 * (prop[k] - base[k])
        lo, hi = 100 * np.nanpercentile(boot[k], 2.5), 100 * np.nanpercentile(boot[k], 97.5)
        flag = "*" if (lo > 0 or hi < 0) else ""
        print(f"{k:45s}{100*base[k]:10.1f}{100*prop[k]:10.1f}{d:+12.1f}   [{lo:+.1f}, {hi:+.1f}] {flag}")
        lines.append([k, 100 * base[k], 100 * prop[k], d, lo, hi])

    # per-image M
    sel = X[:, idx["base_n_omega"]] >= MIN_OMEGA_PX
    mb = X[sel, idx["base_exact"]] / X[sel, idx["base_n_omega"]]
    mp = X[sel, idx["prop_exact"]] / X[sel, idx["prop_n_omega"]]
    dd = 100 * (mp - mb)
    w = X[sel, idx["base_n_omega"]]
    gain_px = (X[sel, idx["prop_exact"]] - X[sel, idx["base_exact"]])
    order = np.argsort(-gain_px)
    top10 = gain_px[order[: max(1, len(order) // 10)]].sum() / max(gain_px.sum(), 1)
    print(f"\nper-image M ({sel.sum()} images with >= {MIN_OMEGA_PX} overlap pixels):")
    print(f"  proposed better by > 1 point: {(dd > 1).sum()}, worse by > 1 point: {(dd < -1).sum()}, "
          f"within 1 point: {(np.abs(dd) <= 1).sum()}")
    print(f"  median per-image difference: {np.median(dd):+.1f} points; "
          f"share of the net gain in exact pixels from the top 10% of images: {100*top10:.0f}%")
    lines += [["images_better_gt1pt", (dd > 1).sum(), "", "", "", ""],
              ["images_worse_gt1pt", (dd < -1).sum(), "", "", "", ""],
              ["images_within_1pt", (np.abs(dd) <= 1).sum(), "", "", "", ""],
              ["median_per_image_diff_pts", np.median(dd), "", "", "", ""],
              ["top10pct_images_share_of_gain_pct", 100 * top10, "", "", "", ""]]

    # transitions
    tr_keys = [f"tr_{a}_{b}" for a in CATS for b in CATS]
    if all(k in idx for k in tr_keys):
        Tm = np.array([[T[f"tr_{a}_{b}"] for b in CATS] for a in CATS])
        print("\ntransition matrix, rows = baseline category, columns = proposed category (% of overlap pixels):")
        print("            " + "".join(f"{c:>9s}" for c in CATS))
        for i, a in enumerate(CATS):
            print(f"{a:12s}" + "".join(f"{100*Tm[i,j]/Tm.sum():9.2f}" for j in range(5)))
        for i, a in enumerate(CATS[1:], 1):
            lines.append([f"base_{a}_to_exact_pct_of_that_category", 100 * Tm[i, 0] / max(Tm[i].sum(), 1), "", "", "", ""])
        lines.append(["base_exact_to_nonexact_pct", 100 * Tm[0, 1:].sum() / max(Tm[0].sum(), 1), "", "", "", ""])
        for i, a in enumerate(CATS):
            for j, b in enumerate(CATS):
                lines.append([f"transition_{a}_to_{b}_pct_of_omega", 100 * Tm[i, j] / Tm.sum(), "", "", "", ""])

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["quantity", "baseline_or_value", "proposed", "difference_pts", "ci_low", "ci_high"])
        wr.writerows(lines)
    print(f"\nsummary written to {out_csv}")
    _figure(base, prop, dd, fig_path)


def _figure(base, prop, dd, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 8, "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
                         "ytick.color": "black", "axes.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 600})
    cols = {"exact": "#1A9850", "missing": "#F4A11D", "extra": "#7B3294", "mixed": "#D73027", "none": "#404040"}
    names = {"exact": "Exact set", "missing": "Labels missing", "extra": "Extra labels",
             "mixed": "Missing and extra", "none": "No labelled class"}
    fig = plt.figure(figsize=(6.3, 2.7))
    ax = fig.add_axes([0.11, 0.38, 0.42, 0.50])
    bars = [("Baseline", base), ("Proposed", prop)]
    for i, (lab, s) in enumerate(bars):
        left = 0.0
        vals = [s["M"]] + [s[f"share_{c}"] for c in CATS[1:]]
        for c, v in zip(CATS, vals):
            ax.barh(i, 100 * v, left=left, color=cols[c], height=0.55, edgecolor="white", linewidth=0.4)
            if v > 0.06:
                ax.text(left + 50 * v, i, f"{100*v:.1f}", ha="center", va="center", color="black")
            left += 100 * v
    ax.set_yticks([0, 1]); ax.set_yticklabels([b[0] for b in bars]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("Share of validation pixels with two or more labels (%)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cols[c]) for c in CATS], labels=[names[c] for c in CATS],
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, handlelength=1.0, columnspacing=1.0)
    fig.text(0.01, 0.95, "(a) Label-set outcome of overlap pixels", va="top")
    bx = fig.add_axes([0.68, 0.38, 0.30, 0.50])
    d = np.sort(dd)
    bx.axhline(0, color="black", lw=0.6)
    bx.plot(np.arange(1, len(d) + 1) / len(d) * 100, d, color="#2166AC", lw=1.2)
    bx.fill_between(np.arange(1, len(d) + 1) / len(d) * 100, 0, d, where=d > 0, color="#2166AC", alpha=0.25, lw=0)
    bx.fill_between(np.arange(1, len(d) + 1) / len(d) * 100, 0, d, where=d < 0, color="#E66101", alpha=0.25, lw=0)
    bx.set_xlabel("Images, sorted by change (%)"); bx.set_ylabel("Change in per-image M (points)")
    for s_ in ("top", "right"):
        bx.spines[s_].set_visible(False)
    fig.text(0.56, 0.95, "(b) Per-image change, proposed minus baseline", va="top")
    fig.savefig(path)
    print(f"figure written to {path}")


# ----------------------------------------------------------------------------- self-test and entry point
def _selftest():
    rng = np.random.default_rng(0)
    rec = LabelSetRecorder()
    for i in range(30):
        lab = rng.random((19, 60, 80)) < 0.12
        pb = lab ^ (rng.random(lab.shape) < 0.08)
        pp = lab ^ (rng.random(lab.shape) < 0.06)
        rec.add(f"img{i}", lab, pb, pp)
    rec.save("selftest_per_image.csv")
    global MIN_OMEGA_PX
    MIN_OMEGA_PX = 10
    analyse(["selftest_per_image.csv"], "selftest_fig.png", "selftest_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("no CSV given: running a self-test on random masks")
        _selftest()
    else:
        analyse(sys.argv[1:])
