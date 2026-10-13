"""Paper C: runs labelset_errors.py on the stored predictions. Version 1.0, 12 October 2026.
CPU only (about 10-20 min, mostly reading the stored predictions from Drive); no model is run.

labelset_errors.py needs a per-image CSV written by its LabelSetRecorder inside an evaluation
loop. This script is that loop: for every validation image it draws the 19 label masks from the polygons at long side
1024 and reads the stored predictions of the baseline and of our model at the same grid (the final step with TTA, and
the best epoch without TTA as a check), the same inputs as inspection_records.py. It then calls the
analysis of labelset_errors.py, which prints the results and writes the summary CSV and the figure.

Writes to <root>/paperc/:
  labelset_per_image_tta_v1_0.csv, labelset_summary_tta_v1_0.csv, PaperC_Fig_labelset_tta_v1_0.png      (with TTA)
  labelset_per_image_plain_v1_0.csv, labelset_summary_plain_v1_0.csv, PaperC_Fig_labelset_plain_v1_0.png (without TTA)
Needs, in the same folder: labelset_errors.py, pipeline.py and
figs_helpers.py.
Colab (CPU runtime):
  from google.colab import drive; drive.mount('/content/drive')
  !python labelset_run.py
"""
import argparse, importlib.util, io, json, os, sys, time, zipfile
import numpy as np

VERSION = 'labelset_run.py'
PAIRS = {  # tag: (baseline run, our run, kind) - 'final' = final step with TTA, 'run' = best epoch without TTA
    'tta': ('B3-L-s0', 'B3-ovl-L0.1-s0', 'final'),
    'plain': ('B3-L-s0', 'B3-ovl-L0.1-s0', 'run')}


def log(*a): print(*a, flush=True)


def load(name, path):
    if not os.path.exists(path): sys.exit(f'{os.path.basename(path)} not found next to {VERSION}')
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--zip', default=None); ap.add_argument('--data', default=None)
    ap.add_argument('--max-images', type=int, default=0, help='test only')
    a = ap.parse_args(); a.runs = os.path.join(a.root, 'paperc', 'runs'); out = os.path.join(a.root, 'paperc')
    here = os.path.dirname(os.path.abspath(__file__))
    pcf = next((p for p in (os.path.join(here, f) for f in ('pipeline.py', 'pipeline.py', 'pipeline.py'))
                if os.path.exists(p)), None)
    if not pcf: sys.exit('pipeline.py not found next to this script')
    LA = load('labelset', os.path.join(here, 'labelset_errors.py'))
    PC = load('paperc', pcf); FG = load('paperc_figs', os.path.join(here, 'figs_helpers.py')); FG.PC = PC
    idx = json.load(open(os.path.join(a.root, 'paperc', 's2_index.json')))['index']
    val = sorted([m for m in idx if m['split'] == 'validation'], key=lambda m: m['name'])
    if a.max_images: val = val[:a.max_images]
    srcs = {}
    for tag, (rb, ro, kind) in PAIRS.items():
        pair = []
        for run in (rb, ro):
            s = FG.pred_source(os.path.join(a.runs, run), kind)
            if s is None or (kind == 'final' and s[2] != 'TTA'):
                log(f'  {tag}: no stored {"final TTA" if kind == "final" else "best-epoch"} predictions of {run}, left out'); break
            zs = [zipfile.ZipFile(z) for z in s[0]]
            pair.append(({n: z for z in zs for n in z.namelist()}, s[1])); log(f'  {tag}: {run}, {s[2]}, {len(s[0])} zip files')
        if len(pair) == 2: srcs[tag] = pair
    if not srcs: sys.exit('no stored predictions found')
    recs = {tag: LA.LabelSetRecorder() for tag in srcs}; imgs = FG.Images(a); t0 = time.time()

    def read(where, mk, stem, shape):
        z = where.get(stem + '.npz')
        if z is None: return None
        d = np.load(io.BytesIO(z.read(stem + '.npz'))); C_, hh, ww = d['shape1024']
        pr = np.unpackbits(d[mk], axis=-1)[..., :ww].astype(bool)
        return pr if pr.shape == shape else None

    for i, m in enumerate(val):
        W, H = m['W'], m['H']; s = PC.RULES['global_long_side'] / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
        stem = os.path.splitext(m['name'])[0]
        gt = PC.rasterize_window(PC.polygons(json.loads(imgs.read(stem + '.json'))), s, 0, 0, w, h)
        for tag, ((wb, kb), (wo, ko)) in srcs.items():
            pb, po = read(wb, kb, stem, gt.shape), read(wo, ko, stem, gt.shape)
            if pb is None or po is None: log(f'  {m["name"]}: {tag} prediction missing or of another size, left out'); continue
            recs[tag].add(m['name'], gt, pb, po)
        if (i + 1) % 100 == 0: log(f'  {i + 1} / {len(val)} images ({time.time() - t0:.0f} s)')
    for tag, rec in recs.items():
        log(f'\n================ {tag}: {"final models with TTA" if tag == "tta" else "best epoch without TTA"} '
            f'(labels and predictions at long side 1024) ================')
        csv_path = os.path.join(out, f'labelset_per_image_{tag}_v1_0.csv'); rec.save(csv_path)
        LA.analyse([csv_path], fig_path=os.path.join(out, f'PaperC_Fig_labelset_{tag}_v1_0.png'),
                   out_csv=os.path.join(out, f'labelset_summary_{tag}_v1_0.csv'))
    log(f'\ndone ({time.time() - t0:.0f} s). Please send the printed output and the files labelset_summary_*_v1_0.csv.')


if __name__ == '__main__':
    main()
