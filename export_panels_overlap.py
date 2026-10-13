"""Paper C: raw panels for more case-study figures (labels, predictions, label-set errors). Version 1.2, 12 October 2026.
CPU only (about 10-15 min); no model is run, the stored predictions are used. Small enough to upload (about 10-20 MB).

The images are chosen by fixed rules on the labels alone, never on the predictions, so the figures are not chosen by
result. The labels of all validation images are drawn at long side 1024 and the images are ranked:
  rebar     the --n-rebar images (default 6) with the most pixels labelled with both ExposedRebars and Rust, leaving
            out the images already shown in Figure 7
  overlap   the --n-overlap images (default 12) with the most pixels carrying two or more labels, leaving out the
            images of the rebar group and those already shown in Figure 7 (val 0375, 0381, 0039)
For each image: <stem>_1024.jpg (JPEG quality 95, no chroma subsampling) and <stem>_masks.npz (19-class masks at long
side 1024, bits packed along the width: labels, B1, baseline, ours, stage2), as in panels_v1_1, so the figure script
reads both. The manifest records the rules, the ranking values and the chosen images.
Writes <root>/paperc/panels_v1_2/ and <root>/paperc/panels_v1_2.zip. Upload the zip.
Needs pipeline.py and figs_helpers.py in the same folder (already on Drive).
Colab (CPU runtime is enough):
  from google.colab import drive; drive.mount('/content/drive')
  !python export_panels_overlap.py
"""
import argparse, importlib.util, json, os, shutil, sys, time
import numpy as np

VERSION = 'export_panels_overlap.py'
KEY = {'B1-s0': 'B1', 'B3-L-s0': 'baseline', 'B3-ovl-L0.1-s0': 'ours', 'Full-on-B3-ovl-L0.1-s0-skelT-s0': 'stage2'}
IN_FIGURE_7 = {'dacl10k_v2_validation_0375.jpg', 'dacl10k_v2_validation_0381.jpg', 'dacl10k_v2_validation_0039.jpg'}


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
    ap.add_argument('--n-rebar', type=int, default=6); ap.add_argument('--n-overlap', type=int, default=12)
    a = ap.parse_args(); a.runs = os.path.join(a.root, 'paperc', 'runs'); t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    pcf = next((p for p in (os.path.join(here, f) for f in ('pipeline.py', 'pipeline.py', 'pipeline.py')) if os.path.exists(p)), None)
    if not pcf: sys.exit('pipeline.py not found next to this script')
    if not os.path.exists(os.path.join(here, 'figs_helpers.py')): sys.exit('figs_helpers.py not found next to this script')
    PC = load('paperc', pcf); FG = load('paperc_figs', os.path.join(here, 'figs_helpers.py')); FG.PC = PC
    out = os.path.join(a.root, 'paperc', 'panels_v1_2'); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    idx = json.load(open(os.path.join(a.root, 'paperc', 's2_index.json')))['index']
    val = sorted([m for m in idx if m['split'] == 'validation'], key=lambda m: m['name'])
    imgs = FG.Images(a); er, ru = PC.CI['ExposedRebars'], PC.CI['Rust']; L = PC.RULES['global_long_side']
    log(f'{VERSION}: ranking {len(val)} validation images by their labels at long side {L}')
    rank, cache = [], {}
    for i, m in enumerate(val):
        W, H = m['W'], m['H']; s = L / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
        gt = PC.rasterize_window(PC.polygons(json.loads(imgs.read(os.path.splitext(m['name'])[0] + '.json'))), s, 0, 0, w, h)
        rank.append({'name': m['name'], 'rebar_rust_px': int((gt[er] & gt[ru]).sum()), 'overlap_px': int((gt.sum(0) >= 2).sum())})
        if (i + 1) % 200 == 0: log(f'  {i + 1} / {len(val)} images ({time.time() - t0:.0f} s)')
    by_rebar = sorted([r for r in rank if r['rebar_rust_px'] > 0 and r['name'] not in IN_FIGURE_7], key=lambda r: -r['rebar_rust_px'])
    rebar = by_rebar[:a.n_rebar]; taken = {r['name'] for r in rebar} | IN_FIGURE_7
    overlap = [r for r in sorted(rank, key=lambda r: -r['overlap_px']) if r['name'] not in taken and r['overlap_px'] > 0][:a.n_overlap]
    man = {'code': VERSION, 'classes': list(PC.CLASSES),
           'rules': {'rebar': f'the {a.n_rebar} validation images with the most pixels labelled ExposedRebars and Rust (long side {L})',
                     'overlap': f'the {a.n_overlap} images with the most pixels carrying two or more labels, without the rebar group '
                                'and the images of Figure 7'},
           'qualifying': {'rebar': len(by_rebar), 'overlap': sum(r['overlap_px'] > 0 for r in rank)},
           'groups': {'rebar': rebar, 'overlap': overlap}, 'images': []}
    log('  rebar group: ' + ', '.join(f'{r["name"][-8:-4]} ({r["rebar_rust_px"]:,} px)' for r in rebar))
    log('  overlap group: ' + ', '.join(f'{r["name"][-8:-4]} ({r["overlap_px"]:,} px)' for r in overlap))
    srcs = {}
    for _, run, kind in FG.MODELS:
        src = FG.pred_source(os.path.join(a.runs, run), kind)
        if src: srcs[run] = src
        else: log(f'  no saved predictions for {run}: left out')
    man['models'] = {KEY[r]: {'run': r, 'tag': s[2], 'mask_key': s[1]} for r, s in srcs.items()}
    meta = {m['name']: m for m in val}
    for r in rebar + overlap:
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
                              'group': 'rebar' if r in rebar else 'overlap'})
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    z = shutil.make_archive(out, 'zip', out)
    log(f'\nwritten {z}: {os.path.getsize(z) / 1e6:.1f} MB ({time.time() - t0:.0f} s). Please upload this zip file.')


if __name__ == '__main__':
    main()
