"""Paper C: every manuscript figure and table from the results saved on Drive. Version 1.2, 12 October 2026
(from v1.1: every figure at 600 dpi; new method_inputs/ folder with the real panels as plain full-resolution PNGs
for the methodology figures: image at long side min(2048, native), labels and every model at long side 1024).
CPU only (no model is run). One command makes all of the following in <root>/paperc/manuscript_figures_v1_2/ (a part whose input
is missing is skipped and named in the log):
  training_curves_v1_2.png / .csv        hold-out mIoU and T mIoU per epoch of every run
  per_class_iou_v1_2.png / .csv          validation IoU per class (full size) of every model; the csv also has
                                         512 x 512 and toolkit scores
  summary_v1_2.csv                       one row per model: mIoU (full size, 512, toolkit), T, rare, clDice, overlap
  bootstrap_<A>_vs_<B>_<tag>_v1_0.png    forest plot of every paired bootstrap found (bootstrap.py output)
  zero_shot_v1_2.png / .csv              S2DS and DeepCrack results (paperc_xdata_v1_x.py output)
  segmentation_overlap_v1_2.png          image | labels | B1 | baseline | ours | ours + stage 2
                                         (Spalling, Rust, ExposedRebars)
  segmentation_thin_v1_2.png             the same images, Crack and Cavity
  overlap_correctness_v1_2.png           pixels with two or more labels: exact label set predicted or not
  rows/<image>_<group>_v1_2.png          one large row per image and class group
  method_inputs/<image>_<panel>.png      exact pixels, no resampling: image_native, image1024, and for each group
                                         (overlap, thin) labels, B1, baseline, ours, stage2 overlays at long side 1024
Images for the segmentation figures (fixed rule, no picking by result): validation images whose labels contain
ExposedRebars with Rust or Spalling and also Crack or Cavity, in name order; --names chooses other images.
Colours: validated reference palette; every set used in one panel passes the all-pairs colour-blind check.

  from google.colab import drive; drive.mount('/content/drive')
  !python figs_helpers.py
"""
import argparse, csv, glob, importlib.util, io, json, os, sys, traceback, zipfile
import numpy as np

VERSION = 'figs_helpers.py'
MODELS = [('B1 (dacl10k paper model)\nFPN + EfficientNet-B4', 'B1-s0', 'run'),
          ('Baseline\nViT-L, no overlap loss', 'B3-L-s0', 'final'),
          ('Ours\nViT-L + overlap loss', 'B3-ovl-L0.1-s0', 'final'),
          ('Ours +\nstage 2', 'Full-on-B3-ovl-L0.1-s0-skelT-s0', 'run')]
# two class groups, each a palette that passes the all-pairs colour check (classes in one panel can touch each other)
GROUPS = {'overlap': [('Spalling', '#2a78d6', 0.45), ('Rust', '#eb6834', 0.45), ('ExposedRebars', '#1baf7a', 0.85)],
          'thin': [('Crack', '#eda100', 1.0), ('Cavity', '#e87ba4', 1.0)]}
GOOD, BAD = '#0ca30c', '#d03b3b'                                         # status colours (correct / wrong)
RUNS = [('B1', 'B1: FPN + EfficientNet-B4', 'B1-s0'), ('base', 'Baseline: ViT-L, no overlap loss', 'B3-L-s0'),
        ('ours', 'Ours: ViT-L + overlap loss', 'B3-ovl-L0.1-s0'),
        ('stage2', 'Ours + stage 2', 'Full-on-B3-ovl-L0.1-s0-skelT-s0')]
STYLE = {'B1': ('#52514e', '--'), 'base': ('#eb6834', '-'), 'ours': ('#2a78d6', '-'), 'stage2': ('#1baf7a', '-')}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
PC = None


def log(*a): print(*a, flush=True)


def hex2rgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


class Images:
    """images and annotation files from the unpacked folder if present, else straight from the dataset zip."""
    def __init__(self, a):
        self.dir = next((d for d in (a.data, '/content/dacl10k/v2', '/content/dacl10k') if d and os.path.isdir(d)), None)
        self.zip = None
        if not self.dir:
            z = a.zip or os.path.join(a.root, 'dacl10k', 'dacl10k_v2_devphase.zip')
            if not os.path.exists(z): sys.exit(f'dataset not found: no unpacked folder and no zip at {z}')
            self.zip = zipfile.ZipFile(z); self.members = {os.path.basename(n): n for n in self.zip.namelist()
                                                          if '/validation/' in n}

    def read(self, name):
        if self.zip: return self.zip.read(self.members[name])
        hits = glob.glob(os.path.join(self.dir, '**', 'validation', name), recursive=True)
        if not hits: sys.exit(f'{name} not found under {self.dir}')
        return open(hits[0], 'rb').read()


def load_rgb(data, W, H, w, h):
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data))
    if im.size != (W, H) and im.size == (H, W): im = ImageOps.exif_transpose(im)
    return np.asarray(im.convert('RGB').resize((w, h), Image.BILINEAR))


def pred_source(rdir, kind):
    """(zip files, mask key) of a run's saved validation predictions, or None."""
    if kind == 'final' and os.path.exists(os.path.join(rdir, 'final_v1_8.json')):
        chosen = json.load(open(os.path.join(rdir, 'final_v1_8.json')))['weights']['chosen']
        zs = sorted(glob.glob(os.path.join(rdir, 'final_v1_8', f'validation_tta_{chosen}', 'preds_*.zip')))
        if zs: return zs, 'mask1024_fixed', 'TTA'
    zs = sorted(glob.glob(os.path.join(rdir, 'eval_validation', 'preds_*.zip')))
    return (zs, 'mask1024', 'no TTA') if zs else None


def load_mask(src, name):
    zs, key, _ = src; member = os.path.splitext(name)[0] + '.npz'
    for zp in zs:
        with zipfile.ZipFile(zp) as z:
            if member in z.namelist():
                d = np.load(io.BytesIO(z.read(member))); C, h, w = d['shape1024']
                return np.unpackbits(d[key], axis=-1)[..., :w].astype(bool)
    return None


def overlay(img, masks, show):
    """image dimmed to grey, then the shown classes in fixed order (ExposedRebars on top of Spalling and Rust)."""
    from scipy.ndimage import binary_dilation
    g = img.astype(np.float32).mean(2, keepdims=True) / 255; out = np.repeat(0.35 + 0.55 * g, 3, 2)
    for cls, col, al in show:
        m = masks[PC.CI[cls]]
        if cls in ('Crack', 'Cavity'): m = binary_dilation(m, iterations=1)   # one pixel wider so thin damage stays visible
        out[m] = (1 - al) * out[m] + al * hex2rgb(col)
        if cls == 'ExposedRebars':
            edge = binary_dilation(m, iterations=2) & ~m                 # outline
            out[edge] = hex2rgb(col) * 0.6
    return np.clip(out, 0, 1)


def correctness(img, gt, pr):
    g = img.astype(np.float32).mean(2, keepdims=True) / 255; out = np.repeat(0.3 + 0.5 * g, 3, 2)
    multi = gt.sum(0) >= 2; ok = (pr == gt).all(0) & multi
    out[ok] = hex2rgb(GOOD); out[multi & ~ok] = hex2rgb(BAD)
    return out, int(multi.sum()), int(ok.sum())


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f: w = csv.writer(f); w.writerow(header); w.writerows(rows)


def style_axes(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2)


def results_of(rdir):
    """validation results: the v1.8 final (TTA) if present, else the best epoch of the run (no TTA)."""
    for f, tag in (('results_validation_v1_8_final_fixed.json', 'TTA'), ('results_validation.json', 'no TTA')):
        p = os.path.join(rdir, f)
        if os.path.exists(p): return json.load(open(p)), tag
    return None, None


def fig_curves(a, out, plt):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8)); rows = []; n = 0
    for key, label, run in RUNS:
        p = os.path.join(a.runs, run, 'history.json')
        if not os.path.exists(p): log(f'  curves: no history for {run}'); continue
        h = [r for r in json.load(open(p)) if r.get('holdout_mIoU_1024') is not None]; n += 1
        col, ls = STYLE[key]
        for ax, k in zip(axs, ('holdout_mIoU_1024', 'holdout_T_mIoU_1024')):
            x = [r['epoch'] for r in h]; y = [r[k] for r in h]
            ax.plot(x, y, ls, color=col, lw=1.8, marker='o', ms=3, label=label)
            ax.annotate(f'{y[-1]:.3f}', (x[-1], y[-1]), xytext=(4, 0), textcoords='offset points', fontsize=6.5, color=INK2, va='center')
        rows += [[run, r['epoch'], r.get('loss'), r['holdout_mIoU_1024'], r['holdout_T_mIoU_1024'], r.get('minutes')] for r in h]
    if not n: plt.close(fig); return False
    for ax, t in zip(axs, ('(a) Hold-out mIoU (long side 1024)', '(b) Hold-out T-class mIoU')):
        ax.set_title(t, loc='left', fontsize=8.5, fontweight='bold'); ax.set_xlabel('epoch of the run'); style_axes(ax)
        from matplotlib.ticker import MaxNLocator; ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].legend(fontsize=6.5, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'training_curves_v1_2.png'), facecolor='white'); plt.close(fig)
    write_csv(os.path.join(out, 'training_curves_v1_2.csv'), ['run', 'epoch', 'loss', 'holdout_mIoU_1024', 'holdout_T_mIoU_1024', 'minutes'], rows)
    return True


def fig_per_class(a, out, plt):
    res = []
    for key, label, run in RUNS:
        r, tag = results_of(os.path.join(a.runs, run))
        if r: res.append((key, f'{label} ({tag})', run, tag, r))
        else: log(f'  per-class: no validation results for {run}')
    if not res: return False
    summ, rows = [], []
    for key, label, run, tag, r in res:
        g = lambda k, f='mIoU': (r.get(k) or {}).get(f)
        summ.append([run, tag, g('iu_native'), g('iu_native', 'T_mIoU'), g('iu_native', 'rare_mIoU'), g('iu512'), g('iu512_toolkit'),
                     (r.get('clDice') or {}).get('Crack'), r.get('multilabel_exact_match')])
        for c in PC.CLASSES:
            rows.append([run, tag, c] + [(r.get(k) or {}).get('per_class', {}).get(c) for k in ('iu_native', 'iu512', 'iu512_toolkit')])
    write_csv(os.path.join(out, 'summary_v1_2.csv'), ['run', 'TTA', 'mIoU_full', 'T_mIoU_full', 'rare_mIoU_full', 'mIoU_512',
              'mIoU_toolkit', 'clDice_crack', 'exact_label_set_match'], summ)
    write_csv(os.path.join(out, 'per_class_iou_v1_2.csv'), ['run', 'TTA', 'class', 'IoU_full', 'IoU_512', 'IoU_toolkit'], rows)
    show = [x for x in res if x[0] != 'stage2']                          # stage 2 changes only the T classes (own figure)
    ref = next((x for x in show if x[0] == 'ours'), show[-1])[4]['iu_native']['per_class']
    cls = sorted(PC.CLASSES, key=lambda c: ref.get(c) or 0)
    fig, ax = plt.subplots(figsize=(7.2, 6.2)); k = len(show); bh = 0.8 / k
    for i, (key, label, run, tag, r) in enumerate(show):
        v = [r['iu_native']['per_class'].get(c) or 0 for c in cls]
        ax.barh(np.arange(len(cls)) + (i - (k - 1) / 2) * bh, v, height=bh * 0.85, color=STYLE[key][0], label=label)
    ax.set_yticks(range(len(cls))); ax.set_yticklabels([c + (' (T)' if c in PC.RULES['T_classes'] else '') for c in cls])
    ax.set_xlabel('IoU, validation, full image size'); style_axes(ax); ax.grid(axis='y', visible=False)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'per_class_iou_v1_2.png'), facecolor='white'); plt.close(fig)
    return True


def fig_bootstrap(a, out, plt):
    files = sorted(glob.glob(os.path.join(a.root, 'paperc', 'bootstrap_*_v1_0.json')))
    for f in files:
        b = json.load(open(f)); m = b['metrics']; names = list(m)
        fig, axs = plt.subplots(1, 2, figsize=(7.2, max(3.2, 0.24 * max(len(names), 19) + 0.8)), gridspec_kw={'wspace': 0.9})
        for ax, items, title in ((axs[0], [(n, m[n]) for n in names], '(a) Metrics'),
                                 (axs[1], sorted(b.get('per_class', {}).get('native', {}).items(), key=lambda kv: -kv[1]['diff']),
                                  '(b) Per-class IoU, full size')):
            for i, (n, v) in enumerate(items):
                y = len(items) - 1 - i; lo, hi = v['ci95']; sig = lo > 0
                ax.plot([lo * 100, hi * 100], [y, y], color='#2a78d6', lw=1.5)
                ax.plot(v['diff'] * 100, y, 'o', ms=5, mfc='#2a78d6' if sig else 'white', mec='#2a78d6', mew=1.3, zorder=3)
            ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _ in items][::-1], fontsize=6.5)
            ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold'); ax.set_xlabel(f'{b["A"]} minus {b["B"]} (points)', fontsize=6.5)
            style_axes(ax); ax.grid(axis='y', visible=False)
        fig.text(0.01, 0.0, f'95% paired-bootstrap intervals, {b["images"]} images, {b["resamples"]} resamples ({b["results"]} results). '
                 'Filled dot: interval above zero.', fontsize=6.5, color=INK2)
        name = os.path.basename(f).replace('.json', '.png')
        fig.savefig(os.path.join(out, name), facecolor='white', bbox_inches='tight'); plt.close(fig)
    return bool(files)


def fig_zero_shot(a, out, plt):
    files = sorted(glob.glob(os.path.join(a.root, 'paperc', 'xdata_v1_0', '*.json')))
    if not files: return False
    res = {}; rows = []
    for f in files:
        r = json.load(open(f)); res[(r['dataset'], r['run'])] = r; t = r['test']
        rows.append([r['dataset'], r['run'], r['chosen_L'], json.dumps(t['IoU']), json.dumps(t.get('crack_ltIoU')), t.get('crack_clDice'),
                     t.get('F1'), t.get('precision'), t.get('recall')])
    write_csv(os.path.join(out, 'zero_shot_v1_2.csv'), ['dataset', 'run', 'input_long_side', 'IoU', 'crack_ltIoU', 'crack_clDice', 'F1',
              'precision', 'recall'], rows)
    keys = [(k, l, run) for k, l, run in RUNS if any(rk[1] == run for rk in res)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={'width_ratios': [4, 3]})
    panels = [('s2ds', [('Crack (ltIoU 4 px)', lambda t: t['crack_ltIoU']['4']), ('Spalling', lambda t: t['IoU']['spalling']),
                        ('Corrosion', lambda t: t['IoU']['corrosion']), ('Efflorescence', lambda t: t['IoU']['efflorescence'])], '(a) S2DS test'),
              ('deepcrack', [('IoU', lambda t: t['IoU']['crack']), ('F1', lambda t: t['F1']), ('clDice', lambda t: t['crack_clDice'])], '(b) DeepCrack test')]
    for ax, (ds, mets, title) in zip(axs, panels):
        k = len(keys); bw = 0.8 / max(k, 1)
        for i, (key, label, run) in enumerate(keys):
            r = res.get((ds, run))
            if not r: continue
            v = [f(r['test']) or 0 for _, f in mets]
            ax.bar(np.arange(len(mets)) + (i - (k - 1) / 2) * bw, v, width=bw * 0.85, color=STYLE[key][0], label=label)
        ax.set_xticks(range(len(mets))); ax.set_xticklabels([m for m, _ in mets], fontsize=6.5)
        ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold'); style_axes(ax); ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('score (no training on these data)'); axs[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'zero_shot_v1_2.png'), facecolor='white'); plt.close(fig)
    return True


def fig_segmentation(a, out, plt, mpatches):
    idx = json.load(open(os.path.join(a.root, 'paperc', 's2_index.json')))['index']
    val = sorted([m for m in idx if m['split'] == 'validation'], key=lambda m: m['name'])
    if a.names: pick = [m for m in val if m['name'] in set(a.names)]
    else:
        ci = PC.CI; has = lambda m, c: m['presence'][ci[c]]
        rule = lambda m: has(m, 'ExposedRebars') and (has(m, 'Rust') or has(m, 'Spalling')) and (has(m, 'Crack') or has(m, 'Cavity'))
        pick = [m for m in val if rule(m)][:a.n]
        log(f'  segmentation: {sum(rule(m) for m in val)} validation images qualify; the first {len(pick)} by name are shown')
    if not pick: log('  segmentation: no images chosen'); return False
    models = []
    for label, run, kind in MODELS:
        src = pred_source(os.path.join(a.runs, run), kind)
        if src: models.append((label, src, run))
        else: log(f'  segmentation: no saved predictions for {run}, column skipped')
    imgs = Images(a); rows = []; export = []
    for m in pick:
        W, H = m['W'], m['H']; s = PC.RULES['global_long_side'] / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
        img = load_rgb(imgs.read(m['name']), W, H, w, h)
        gt = PC.rasterize_window(PC.polygons(json.loads(imgs.read(os.path.splitext(m['name'])[0] + '.json'))), s, 0, 0, w, h)
        preds = [(label, load_mask(src, m['name']), run) for label, src, run in models]
        native = load_rgb(imgs.read(m['name']), W, H, *(lambda t: (max(1, round(W * t)), max(1, round(H * t))))(min(1.0, 2048 / max(W, H))))
        rows.append((m['name'], img, gt, [(l, p) for l, p, _ in preds if p is not None and p.shape == gt.shape]))
        export.append((m['name'], img, native, gt, [(r, p) for _, p, r in preds if p is not None and p.shape == gt.shape]))
    titles = ['Image', 'Labels'] + [l for l, _ in rows[0][3]]
    os.makedirs(os.path.join(out, 'rows'), exist_ok=True)
    for group, show in GROUPS.items():
        leg = [mpatches.Patch(color=c, label=n) for n, c, _ in show]
        ncol = len(titles); fig, axs = plt.subplots(len(rows), ncol, figsize=(1.25 * ncol, 1.0 * len(rows) + 0.6), squeeze=False)
        for r, (name, img, gt, preds) in enumerate(rows):
            panels = [img / 255, overlay(img, gt, show)] + [overlay(img, p, show) for _, p in preds]
            for c, pnl in enumerate(panels):
                ax = axs[r, c]; ax.imshow(pnl); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
                if r == 0: ax.set_title(titles[c], fontsize=6.5)
            axs[r, 0].set_ylabel(name.replace('dacl10k_v2_validation_', 'val '), fontsize=6)
        fig.legend(handles=leg, loc='lower center', ncol=len(show), frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.2, h_pad=0.2)
        fig.savefig(os.path.join(out, f'segmentation_{group}_v1_2.png'), facecolor='white', bbox_inches='tight'); plt.close(fig)
        for name, img, gt, preds in rows:
            panels = [img / 255, overlay(img, gt, show)] + [overlay(img, p, show) for _, p in preds]
            fig, axs2 = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.8))
            for ax, pnl, t in zip(axs2, panels, titles): ax.imshow(pnl); ax.set_title(t, fontsize=9); ax.axis('off')
            fig.legend(handles=leg, loc='lower center', ncol=len(show), frameon=False, fontsize=9)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            fig.savefig(os.path.join(out, 'rows', f'{os.path.splitext(name)[0]}_{group}_v1_2.png'), dpi=600, facecolor='white'); plt.close(fig)
    fig, axs = plt.subplots(len(rows), 1 + len(rows[0][3]), figsize=(1.25 * (1 + len(rows[0][3])), 1.2 * len(rows) + 0.6), squeeze=False)
    for r, (name, img, gt, preds) in enumerate(rows):
        axs[r, 0].imshow(img / 255)
        for c, (label, p) in enumerate(preds, 1):
            pnl, n, k = correctness(img, gt, p); axs[r, c].imshow(pnl)
            axs[r, c].set_xlabel(f'{100 * k / max(n, 1):.0f}% of {n} px correct', fontsize=5.5)
            if r == 0: axs[r, c].set_title(label, fontsize=6.5)
        if r == 0: axs[r, 0].set_title('Image', fontsize=6.5)
        axs[r, 0].set_ylabel(name.replace('dacl10k_v2_validation_', 'val '), fontsize=6)
        for ax in axs[r]: ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.legend(handles=[mpatches.Patch(color=GOOD, label='exact label set predicted'), mpatches.Patch(color=BAD, label='wrong label set')],
               loc='lower center', ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 1.0, 'Pixels with two or more labels only', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), w_pad=0.2, h_pad=1.0)
    fig.savefig(os.path.join(out, 'overlap_correctness_v1_2.png'), facecolor='white', bbox_inches='tight'); plt.close(fig)

    # exact-pixel panels for the methodology figures (PIL, no matplotlib resampling)
    from PIL import Image as PILImage
    key = {'B1-s0': 'B1', 'B3-L-s0': 'baseline', 'B3-ovl-L0.1-s0': 'ours', 'Full-on-B3-ovl-L0.1-s0-skelT-s0': 'stage2'}
    md = os.path.join(out, 'method_inputs'); os.makedirs(md, exist_ok=True)
    for name, img, native, gt, preds in export:
        stem = os.path.splitext(name)[0].replace('dacl10k_v2_validation_', 'val')
        PILImage.fromarray(native).save(os.path.join(md, f'{stem}_image_native.png'))
        PILImage.fromarray(img).save(os.path.join(md, f'{stem}_image1024.png'))
        for group, show in GROUPS.items():
            for k, msk in [('labels', gt)] + [(key.get(r, r), p) for r, p in preds]:
                PILImage.fromarray((overlay(img, msk, show) * 255).round().astype(np.uint8)).save(os.path.join(md, f'{stem}_{group}_{k}.png'))
    log(f'  method inputs: {len(os.listdir(md))} files in {md}')
    log('  segmentation images: ' + ', '.join(n for n, *_ in rows))
    return True


def main():
    global PC
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--zip', default=None); ap.add_argument('--data', default=None)
    ap.add_argument('--names', nargs='*', default=None, help='validation image names instead of the fixed rule')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--paperc', default=None)
    a = ap.parse_args(); a.runs = os.path.join(a.root, 'paperc', 'runs')
    spec = importlib.util.spec_from_file_location('paperc', a.paperc or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline.py'))
    PC = importlib.util.module_from_spec(spec); spec.loader.exec_module(PC)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    plt.rcParams.update({'font.size': 7.5, 'savefig.dpi': 600, 'font.family': 'serif', 'mathtext.fontset': 'stix',
                         'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif']})
    out = os.path.join(a.root, 'paperc', 'manuscript_figures_v1_2'); os.makedirs(out, exist_ok=True)   # own folder: never overwrites paperc_segfig output
    log(f'{VERSION} | output {out}')
    parts = [('training curves', lambda: fig_curves(a, out, plt)), ('per-class IoU and summary', lambda: fig_per_class(a, out, plt)),
             ('bootstrap plots', lambda: fig_bootstrap(a, out, plt)), ('zero-shot results', lambda: fig_zero_shot(a, out, plt)),
             ('segmentation images', lambda: fig_segmentation(a, out, plt, mpatches))]
    for name, fn in parts:
        try: log(f'{name}: ' + ('done' if fn() else 'skipped (input not found)'))
        except Exception: log(f'{name}: FAILED (the other parts continue)'); traceback.print_exc()
    log(f'\nall files are in {out}:'); [log('  ' + f) for f in sorted(os.listdir(out))]


if __name__ == '__main__':
    main()
