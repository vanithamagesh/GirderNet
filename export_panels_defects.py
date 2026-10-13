"""Paper C: raw panels for many more segmentation figures, one group per inspection defect. Version 1.3, 12 October 2026.
CPU only (about 10-15 min); no model is run, the stored predictions are used. About 20-25 MB to upload.

The images are chosen by fixed rules on the labels alone, never on the predictions, so the figures are not chosen by
result. The labels of all validation images are drawn at long side 1024 and, leaving out the 34 images exported by
panels_v1_1 and panels_v1_2, the images are ranked per inspection defect (the definitions of
inspection_records.py):
  one group per defect   the --n-per-defect images (default 6) with the most labelled pixels of that defect
                         (exposed rebar with rust; exposed rebar without rust; spall with rust staining; spall without
                         rust; crack with efflorescence; wet spot on weathered concrete; rust at a drainage outlet or joint)
  typical                --n-typical images (default 12) drawn at random (seed 0) from the remaining images with at
                         least one labelled defect, to show ordinary images and not only the largest defects
Each image is used once, in the first group that takes it. For each image: <stem>_1024.jpg (JPEG quality 95, no chroma
subsampling) and <stem>_masks.npz (19-class masks at long side 1024, bits packed along the width: labels, B1,
baseline, ours, stage2), as in panels_v1_1 and v1_2, so the figure script reads all three.
Writes <root>/paperc/panels_v1_3/ and <root>/paperc/panels_v1_3.zip. Upload the zip.
Needs pipeline.py and figs_helpers.py in the same folder (already on Drive).
Colab (CPU runtime is enough):
  from google.colab import drive; drive.mount('/content/drive')
  !python export_panels_defects.py
"""
import argparse, importlib.util, json, os, shutil, sys, time
import numpy as np

VERSION = 'export_panels_defects.py'
KEY = {'B1-s0': 'B1', 'B3-L-s0': 'baseline', 'B3-ovl-L0.1-s0': 'ours', 'Full-on-B3-ovl-L0.1-s0-skelT-s0': 'stage2'}
EXPORTED = {f'dacl10k_v2_validation_{k}.jpg' for k in (           # panels_v1_1 and panels_v1_2
    '0016 0017 0018 0021 0024 0031 0039 0045 0046 0060 0066 0067 0237 0295 0375 0381 '
    '0034 0104 0105 0108 0111 0220 0310 0362 0371 0373 0460 0589 0592 0622 0871 0922 0946 0968').split()}
RECORDS = [  # key, description, groups (one class of each group present), excluded classes: as paperc_inspection_records_v1_0
    ('rebar_rust', 'exposed rebar with rust', [['ExposedRebars'], ['Rust']], []),
    ('rebar_no_rust', 'exposed rebar without rust', [['ExposedRebars']], ['Rust']),
    ('spall_rust', 'spall with rust staining, no exposed rebar', [['Spalling'], ['Rust']], ['ExposedRebars']),
    ('spall_plain', 'spall without rust or exposed rebar', [['Spalling']], ['Rust', 'ExposedRebars']),
    ('crack_efflo', 'crack with efflorescence', [['Crack', 'ACrack'], ['Efflorescence']], []),
    ('wet_weathered', 'wet spot on weathered concrete', [['Wetspot'], ['Weathering']], []),
    ('rust_fixture', 'rust at a drainage outlet or joint', [['Drainage', 'EJoint'], ['Rust']], [])]
MIN_FRAC = 0.0005                      # a defect counts as labelled if it covers at least 0.05% of the image


def log(*a): print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def save_jpg(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path, quality=95, subsampling=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--zip', default=None); ap.add_argument('--data', default=None)
    ap.add_argument('--n-per-defect', type=int, default=6); ap.add_argument('--n-typical', type=int, default=12)
    a = ap.parse_args(); a.runs = os.path.join(a.root, 'paperc', 'runs'); t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    pcf = next((p for p in (os.path.join(here, f) for f in ('pipeline.py', 'pipeline.py', 'pipeline.py')) if os.path.exists(p)), None)
    if not pcf: sys.exit('pipeline.py not found next to this script')
    if not os.path.exists(os.path.join(here, 'figs_helpers.py')): sys.exit('figs_helpers.py not found next to this script')
    PC = load('paperc', pcf); FG = load('paperc_figs', os.path.join(here, 'figs_helpers.py')); FG.PC = PC
    out = os.path.join(a.root, 'paperc', 'panels_v1_3'); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    idx = json.load(open(os.path.join(a.root, 'paperc', 's2_index.json')))['index']
    val = sorted([m for m in idx if m['split'] == 'validation'], key=lambda m: m['name'])
    imgs = FG.Images(a); L = PC.RULES['global_long_side']
    log(f'{VERSION}: ranking {len(val)} validation images by their labels at long side {L}')
    rank, cache = [], {}
    for i, m in enumerate(val):
        W, H = m['W'], m['H']; s = L / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
        gt = PC.rasterize_window(PC.polygons(json.loads(imgs.read(os.path.splitext(m['name'])[0] + '.json'))), s, 0, 0, w, h)
        r = {'name': m['name'], 'px': w * h, 'overlap_px': int((gt.sum(0) >= 2).sum())}
        for key, _, groups, excl in RECORDS:
            mk = np.ones((h, w), bool)
            for g in groups: mk &= np.any(gt[[PC.CI[c] for c in g]], 0)
            for c in excl: mk &= ~gt[PC.CI[c]]
            r[key] = int(mk.sum())
        rank.append(r)
        if (i + 1) % 200 == 0: log(f'  {i + 1} / {len(val)} images ({time.time() - t0:.0f} s)')
    taken, groups = set(EXPORTED), {}
    for key, desc, _, _ in RECORDS:
        cand = sorted([r for r in rank if r['name'] not in taken and r[key] >= MIN_FRAC * r['px']], key=lambda r: -r[key])
        groups[key] = cand[:a.n_per_defect]; taken |= {r['name'] for r in groups[key]}
        log(f'  {desc}: {len(cand)} images qualify; ' + ', '.join(f'{r["name"][-8:-4]} ({100 * r[key] / r["px"]:.1f}%)' for r in groups[key]))
    pool = sorted([r for r in rank if r['name'] not in taken and any(r[k] >= MIN_FRAC * r['px'] for k, *_ in RECORDS)],
                  key=lambda r: r['name'])
    pick = np.random.default_rng(0).choice(len(pool), size=min(a.n_typical, len(pool)), replace=False)
    groups['typical'] = [pool[i] for i in sorted(pick)]
    log(f'  typical: {len(pool)} images qualify; ' + ', '.join(r['name'][-8:-4] for r in groups['typical']))
    man = {'code': VERSION, 'classes': list(PC.CLASSES),
           'rules': {**{key: f'the {a.n_per_defect} images with the most labelled pixels of "{desc}" (long side {L}), '
                             'not exported before and not taken by an earlier group' for key, desc, _, _ in RECORDS},
                     'typical': f'{a.n_typical} images drawn at random (seed 0) from the remaining images with a labelled defect'},
           'groups': groups, 'images': []}
    chosen = [(g, r) for g, rs in groups.items() for r in rs]
    srcs = {}
    for _, run, kind in FG.MODELS:
        src = FG.pred_source(os.path.join(a.runs, run), kind)
        if src: srcs[run] = src
        else: log(f'  no saved predictions for {run}: left out')
    man['models'] = {KEY[r]: {'run': r, 'tag': s[2], 'mask_key': s[1]} for r, s in srcs.items()}
    meta = {m['name']: m for m in val}
    for g, r in chosen:
        m = meta[r['name']]; W, H = m['W'], m['H']; s = L / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
        stem = os.path.splitext(m['name'])[0]
        save_jpg(FG.load_rgb(imgs.read(m['name']), W, H, w, h), os.path.join(out, f'{stem}_1024.jpg'))
        gt = PC.rasterize_window(PC.polygons(json.loads(imgs.read(stem + '.json'))), s, 0, 0, w, h)
        arrays = {'labels': np.packbits(np.asarray(gt, bool), axis=-1)}
        for run, src in srcs.items():
            p = FG.load_mask(src, m['name'])
            if p is None or p.shape != gt.shape: log(f'  {m["name"]}: {run} prediction missing or of another size'); continue
            arrays[KEY[run]] = np.packbits(p, axis=-1)
        np.savez_compressed(os.path.join(out, f'{stem}_masks.npz'), shape=np.array(gt.shape), **arrays)
        man['images'].append({'name': m['name'], 'W': W, 'H': H, 'masks': sorted(arrays),
                              'group': g})
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    z = shutil.make_archive(out, 'zip', out)
    log(f'\nwritten {z}: {os.path.getsize(z) / 1e6:.1f} MB ({time.time() - t0:.0f} s). Please upload this zip file.')


if __name__ == '__main__':
    main()
