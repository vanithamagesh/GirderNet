"""Paper C: from label sets to inspection records. Version 1.0, 12 October 2026.
CPU only (about 10-20 min, mostly reading the stored predictions from Drive); no model is run.

An inspector records defects, not pixels. This script turns the label sets of every validation image into a defect
list, the way an inspection record would list them, and compares the list derived from the predictions with the list
derived from the labels. Seven combined defects are defined in advance from the overlap structure of dacl10k
(Table 5 of the manuscript). A pixel belongs to a defect if its label set contains one class of every required group
and none of the excluded classes, for example 'exposed rebar with rust' = ExposedRebars and Rust. A defect is present
in an image if it covers at least 0.05% of the image (sensitivity: 0.01% and 0.2%).

Per source (labels drawn from the polygons at long side 1024; stored mask1024 predictions of each run) it reports
  per defect   image-level precision, recall and F1 of presence; record IoU (pixels, summed over images);
               area ratio (predicted / labelled area, summed over images)
  per image    exact record-list match (predicted list = labelled list, over images with at least one labelled
               defect) and list Jaccard (|common| / |either|, over images where either list is non-empty)
  overall      macro F1 over the defects with at least 5 labelled images
and a paired bootstrap over images (2,000 resamples, seed 0) for ours minus baseline. It also writes the defect
lists of every image (labels and each model) to a CSV, so that example inspection records can be shown in the paper.

Writes <root>/paperc/inspection_records_v1_0.json and inspection_records_v1_0_lists.csv. Needs pipeline.py (or
v1_8) and figs_helpers.py next to this file (both on Drive already).
Colab (CPU runtime):
  from google.colab import drive; drive.mount('/content/drive')
  !python inspection_records.py
"""
import argparse, csv, importlib.util, io, json, os, sys, time, zipfile
import numpy as np

VERSION = 'inspection_records.py'
SOURCES = [   # key, run, kind ('final' = v1.8 final step with TTA, 'run' = best epoch without TTA)
    ('B1', 'B1-s0', 'run'), ('baseline_tta', 'B3-L-s0', 'final'), ('ours_tta', 'B3-ovl-L0.1-s0', 'final'),
    ('baseline_plain', 'B3-L-s0', 'run'), ('ours_plain', 'B3-ovl-L0.1-s0', 'run'),
    ('stage2', 'Full-on-B3-ovl-L0.1-s0-skelT-s0', 'run'), ('lrd_tta', 'B3-lrd-L0.1-s0', 'final')]
PAIRS = [('ours_tta', 'baseline_tta'), ('ours_plain', 'baseline_plain'), ('lrd_tta', 'ours_tta')]
# key, record text, required groups (one class of each group), excluded classes, defect names of the element manual
RECORDS = [
    ('rebar_rust', 'exposed rebar with rust', [['ExposedRebars'], ['Rust']], [],
     'Exposed Rebar; Efflorescence/Rust Staining'),
    ('rebar_no_rust', 'exposed rebar without rust', [['ExposedRebars']], ['Rust'], 'Exposed Rebar'),
    ('spall_rust', 'spall with rust staining, no exposed rebar', [['Spalling'], ['Rust']], ['ExposedRebars'],
     'Delamination/Spall/Patched Area; Efflorescence/Rust Staining'),
    ('spall_plain', 'spall without rust or exposed rebar', [['Spalling']], ['Rust', 'ExposedRebars'],
     'Delamination/Spall/Patched Area'),
    ('crack_efflo', 'crack with efflorescence', [['Crack', 'ACrack'], ['Efflorescence']], [],
     'Cracking; Efflorescence/Rust Staining'),
    ('wet_weathered', 'wet spot on weathered concrete', [['Wetspot'], ['Weathering']], [], 'moisture on a weathered surface'),
    ('rust_fixture', 'rust at a drainage outlet or joint', [['Drainage', 'EJoint'], ['Rust']], [],
     'Efflorescence/Rust Staining at a fixture')]
MIN_FRAC = 0.0005                      # presence: at least 0.05% of the image
SENS_FRAC = (0.0001, 0.002)
MIN_LABELLED_IMAGES = 5                # defects with fewer labelled images are reported but left out of the macro F1
RESAMPLES = 2000
CASE_IMAGES = ('0381', '0375', '0039', '0054')


def log(*a): print(*a, flush=True)


def load(name, path):
    if not os.path.exists(path): sys.exit(f'{os.path.basename(path)} not found next to {VERSION}')
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def record_masks(masks, CI):
    """{record key: (h, w) bool} for (19, h, w) bool label or prediction masks."""
    out = {}
    for key, _, groups, excl, _ in RECORDS:
        m = np.ones(masks.shape[1:], bool)
        for g in groups: m &= np.any(masks[[CI[c] for c in g]], 0)
        for c in excl: m &= ~masks[CI[c]]
        out[key] = m
    return out


def image_stats(gt_rec, pr_rec):
    """per-record pixel counts of one image and one source: labelled, predicted, intersection, union."""
    return {k: (int(gt_rec[k].sum()), int(pr_rec[k].sum()), int((gt_rec[k] & pr_rec[k]).sum()), int((gt_rec[k] | pr_rec[k]).sum()))
            for k in gt_rec}


def metrics(S, npx, frac=MIN_FRAC):
    """metrics from per-image counts S[record] = (n_images, 4) int array and image sizes npx (n_images,)."""
    out = {'per_record': {}}; thr = frac * npx; lab_sets, pr_sets = [], []
    f1s = []
    for key, *_ in RECORDS:
        lab, pr, inter, uni = S[key].T
        lp, pp = lab >= thr, pr >= thr
        tp, fp, fn = int((lp & pp).sum()), int((~lp & pp).sum()), int((lp & ~pp).sum())
        nan = float('nan')                                           # undefined when nothing is labelled or predicted
        prec = tp / (tp + fp) if tp + fp else nan; rec = tp / (tp + fn) if tp + fn else nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan
        out['per_record'][key] = {'labelled_images': int(lp.sum()), 'predicted_images': int(pp.sum()), 'precision': prec,
                                  'recall': rec, 'f1': f1, 'record_iou': inter.sum() / uni.sum() if uni.sum() else nan,
                                  'area_ratio': pr.sum() / max(lab.sum(), 1)}
        if lp.sum() >= MIN_LABELLED_IMAGES: f1s.append(f1)
        lab_sets.append(lp); pr_sets.append(pp)
    L, P = np.stack(lab_sets, 1), np.stack(pr_sets, 1)            # (n_images, n_records) presence
    has_lab = L.any(1); either = L.any(1) | P.any(1)
    exact = (L == P).all(1)
    jac = (L & P).sum(1) / np.maximum((L | P).sum(1), 1)
    out['macro_f1'] = float(np.mean(f1s)) if f1s else float('nan')
    out['exact_list_match'] = float(exact[has_lab].mean()) if has_lab.any() else float('nan')
    out['list_jaccard'] = float(jac[either].mean()) if either.any() else float('nan')
    out['images_with_labelled_defect'] = int(has_lab.sum())
    return out


def flat(m):
    """scalar outcomes of a metrics dict for the bootstrap."""
    r = {'macro_f1': m['macro_f1'], 'exact_list_match': m['exact_list_match'], 'list_jaccard': m['list_jaccard']}
    for k, v in m['per_record'].items(): r[f'{k}_f1'] = v['f1']; r[f'{k}_record_iou'] = v['record_iou']
    return r


def defect_list(stats, npx, which):
    """'record text (x.xx%); ...' for the defects present in one image; which = 0 labels, 1 prediction."""
    txt = {k: t for k, t, *_ in RECORDS}; items = []
    for k, c in stats.items():
        if c[which] >= MIN_FRAC * npx: items.append(f'{txt[k]} ({100 * c[which] / npx:.2f}%)')
    return '; '.join(items) if items else 'none'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--zip', default=None); ap.add_argument('--data', default=None)
    ap.add_argument('--max-images', type=int, default=0, help='test only')
    a = ap.parse_args(); a.runs = os.path.join(a.root, 'paperc', 'runs')
    here = os.path.dirname(os.path.abspath(__file__))
    pcf = next((p for p in (os.path.join(here, 'pipeline.py'), os.path.join(here, 'pipeline.py')) if os.path.exists(p)), None)
    if not pcf: sys.exit('pipeline.py not found next to this script')
    PC = load('paperc', pcf); FG = load('paperc_figs', os.path.join(here, 'figs_helpers.py')); FG.PC = PC
    idx = json.load(open(os.path.join(a.root, 'paperc', 's2_index.json')))['index']
    val = sorted([m for m in idx if m['split'] == 'validation'], key=lambda m: m['name'])
    if a.max_images: val = val[:a.max_images]
    srcs = {}
    for key, run, kind in SOURCES:
        s = FG.pred_source(os.path.join(a.runs, run), kind)
        if s is None: log(f'  {key}: no stored predictions of {run}, left out'); continue
        if kind == 'final' and s[2] != 'TTA': log(f'  {key}: final step of {run} not found, left out'); continue
        zs = [zipfile.ZipFile(z) for z in s[0]]
        srcs[key] = ({n: z for z in zs for n in z.namelist()}, s[1]); log(f'  {key}: {run}, {s[2]}, {len(s[0])} zip files')
    if not srcs: sys.exit('no stored predictions found')
    imgs = FG.Images(a); S = {k: {r[0]: [] for r in RECORDS} for k in srcs}; npx, names, lists = [], [], []; t0 = time.time()
    for i, m in enumerate(val):
        W, H = m['W'], m['H']; s = PC.RULES['global_long_side'] / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
        stem = os.path.splitext(m['name'])[0]
        gt = PC.rasterize_window(PC.polygons(json.loads(imgs.read(stem + '.json'))), s, 0, 0, w, h)
        gt_rec = record_masks(gt, PC.CI); row = {}
        for key, (where, mk) in srcs.items():
            z = where.get(stem + '.npz')
            if z is None: row = None; break
            d = np.load(io.BytesIO(z.read(stem + '.npz'))); C_, hh, ww = d['shape1024']
            pr = np.unpackbits(d[mk], axis=-1)[..., :ww].astype(bool)
            if pr.shape != gt.shape: row = None; break
            row[key] = image_stats(gt_rec, record_masks(pr, PC.CI))
        if row is None: log(f'  {m["name"]}: missing or mismatched prediction, image left out of all sources'); continue
        names.append(m['name']); npx.append(w * h)
        first = next(iter(row.values()))
        lists.append({'image': m['name'], 'labels': defect_list(first, w * h, 0),
                      **{k: defect_list(v, w * h, 1) for k, v in row.items()}})
        for key, st in row.items():
            for rk, c in st.items(): S[key][rk].append(c)
        if (i + 1) % 100 == 0: log(f'  {i + 1} / {len(val)} images ({time.time() - t0:.0f} s)')
    npx = np.array(npx, np.float64); A = {k: {rk: np.array(v, np.int64).reshape(-1, 4) for rk, v in r.items()} for k, r in S.items()}
    out = {'code': VERSION, 'grid': 'long side 1024 (labels drawn from the polygons at this size; stored masks)',
           'images': len(names), 'presence_threshold_fraction': MIN_FRAC, 'resamples': RESAMPLES,
           'records': [{'key': k, 'text': t, 'requires_one_of_each': g, 'excludes': e, 'manual_defects': md} for k, t, g, e, md in RECORDS],
           'summary': {k: metrics(v, npx) for k, v in A.items()},
           'sensitivity': {str(f): {k: {kk: metrics(v, npx, f)[kk] for kk in ('macro_f1', 'exact_list_match', 'list_jaccard')}
                                    for k, v in A.items()} for f in SENS_FRAC},
           'paired': {}, 'case_images': [r for r in lists if any(c in r['image'] for c in CASE_IMAGES)]}
    rng = np.random.default_rng(0); n = len(names); idxs = [rng.integers(0, n, n) for _ in range(RESAMPLES)]
    for ka, kb in PAIRS:
        if ka not in A or kb not in A: continue
        fa, fb = flat(out['summary'][ka]), flat(out['summary'][kb]); boots = {k: [] for k in fa}
        for ii in idxs:
            ra = flat(metrics({rk: A[ka][rk][ii] for rk in A[ka]}, npx[ii]))
            rb = flat(metrics({rk: A[kb][rk][ii] for rk in A[kb]}, npx[ii]))
            for k in fa: boots[k].append(ra[k] - rb[k])
        res = {}
        for k in fa:
            b = np.array(boots[k], float); b = b[np.isfinite(b)]
            res[k] = {'A': round(float(fa[k]), 5), 'B': round(float(fb[k]), 5), 'diff': round(float(fa[k] - fb[k]), 5),
                      'ci95': [round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)] if len(b) else None}
        out['paired'][f'{ka}_minus_{kb}'] = res
    path = os.path.join(a.root, 'paperc', 'inspection_records_v1_0.json'); json.dump(out, open(path, 'w'), indent=1)
    cpath = os.path.join(a.root, 'paperc', 'inspection_records_v1_0_lists.csv')
    with open(cpath, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['image', 'labels'] + list(srcs)); wr.writeheader(); wr.writerows(lists)
    log(f'\n{VERSION}: {len(names)} validation images; defect present if it covers at least {100 * MIN_FRAC:.2f}% of the image')
    for k, m in out['summary'].items():
        log(f'\n  {k}: macro F1 {m["macro_f1"]:.3f} | exact record-list match {m["exact_list_match"]:.3f} '
            f'({m["images_with_labelled_defect"]} images with a labelled defect) | list Jaccard {m["list_jaccard"]:.3f}')
        for rk, v in m['per_record'].items():
            log(f'    {rk:14s} labelled in {v["labelled_images"]:4d} images | P {v["precision"]:.3f} R {v["recall"]:.3f} '
                f'F1 {v["f1"]:.3f} | record IoU {v["record_iou"]:.3f} | area ratio {v["area_ratio"]:.2f}')
    for pk, res in out['paired'].items():
        log(f'\n{pk} (paired bootstrap, 95% interval):')
        for k, v in res.items():
            if v['ci95'] is None: continue
            sig = ' above 0' if v['ci95'][0] > 0 else (' below 0' if v['ci95'][1] < 0 else '')
            log(f'  {k:26s} {v["A"]:.4f} {v["B"]:.4f} {v["diff"]:+.4f} [{v["ci95"][0]:+.4f}, {v["ci95"][1]:+.4f}]{sig}')
    log('\nexample inspection records:')
    for r in out['case_images']:
        log(f'  {r["image"]}\n    labels: {r["labels"]}' + ''.join(f'\n    {k}: {r[k]}' for k in srcs if k in r))
    log(f'\nwritten {path}\nwritten {cpath}')


if __name__ == '__main__':
    main()
