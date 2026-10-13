"""Paper C (dacl10k v2): paired bootstrap over the validation images, run A against run B. Version 1.0, 11 October 2026.

S1 (section 9): paired bootstrap over validation images, 2000 resamples; a difference counts as supported if the whole 95%
interval is above zero. CPU only (no GPU, no images, about a minute): it reads the per-image counts that paperc saved.
Every resample draws the same images for both runs (paired).

  from google.colab import drive; drive.mount('/content/drive')
  !python bootstrap.py --a B3-ovl-L0.1-s0 --b B3-L-s0

Default: the v1.8 final results (per_image_validation_v1_8_final_fixed.npz: chosen weights, TTA, threshold 0.5).
--plain: the best-epoch results of the run step (per_image_validation.npz: no TTA).
Writes <root>/paperc/bootstrap_<A>_vs_<B>_<final|plain>_v1_0.json.
"""
import argparse, json, os, sys, warnings
import numpy as np

VERSION = 'bootstrap.py'
CLASSES = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity', 'Spalling',
           'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 'Bearing', 'EJoint', 'Drainage', 'PEquipment',
           'JTape', 'WConccor']
CI = {c: i for i, c in enumerate(CLASSES)}
TIDX = [CI[c] for c in ('Crack', 'Cavity', 'ExposedRebars', 'Rust')]
RARE = [CI[c] for c in ('WConccor', 'Rockpocket', 'ACrack', 'EJoint', 'Restformwork', 'Bearing')]
PAIRS = ('ExposedRebars&Rust', 'ExposedRebars&Spalling', 'Spalling&Rust')
RESAMPLES, SEED = 2000, 0


def load(rdir, fname):
    p = os.path.join(rdir, fname)
    if not os.path.exists(p): sys.exit(f'missing {p} (run the final step of this run first, or use --plain)')
    z = np.load(p, allow_pickle=True); names = [str(n) for n in z['names']]
    return names, {k: z[k].astype(np.float64) for k in z.files if k != 'names'}


def ratio(num, den):
    with np.errstate(invalid='ignore', divide='ignore'): return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def metrics(W, d):
    """every metric for each row of the image weights W (resamples x images); same formulas as paperc summarise()."""
    out = {}
    for key, tag in (('iu_native', 'native'), ('iu512', '512x512'), ('iu512_toolkit', 'toolkit')):
        if key not in d: continue
        iou = ratio(W @ d[key][:, :, 0], W @ d[key][:, :, 1])                       # (resamples, classes)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)                           # all-nan rows (absent classes)
            out[f'mIoU {tag}'] = np.nanmean(iou, 1); out[f'T mIoU {tag}'] = np.nanmean(iou[:, TIDX], 1)
            if tag == 'native': out['rare mIoU native'] = np.nanmean(iou[:, RARE], 1)
        out[f'per_class {tag}'] = iou
    if 'cldice' in d:
        c = W @ d['cldice'][:, 0, :]                                                  # Crack: [sp&G, sp, sg&P, sg]
        tp, ts = c[:, 0] / np.maximum(c[:, 1], 1), c[:, 2] / np.maximum(c[:, 3], 1)
        out['clDice Crack'] = np.where(tp + ts > 0, 2 * tp * ts / np.where(tp + ts > 0, tp + ts, 1), 0.0)
        m = W @ d['multi']; out['exact label-set match (multi-label pixels)'] = m[:, 1] / np.maximum(m[:, 0], 1)
        for i, n in enumerate(PAIRS):
            pa = W @ d['pairs'][:, i, :]; out[f'pair IoU {n}'] = pa[:, 0] / np.maximum(pa[:, 1], 1)
        for i, n in enumerate(('Cavity', 'Rust', 'ExposedRebars')):
            ins = W @ d['instance'][:, i, :]; out[f'instance recall {n}'] = ins[:, 0] / np.maximum(ins[:, 1], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='run A (e.g. our model B3-ovl-L0.1-s0)')
    ap.add_argument('--b', required=True, help='run B (e.g. the baseline B3-L-s0)')
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--plain', action='store_true', help='best-epoch results without TTA instead of the v1.8 final')
    ap.add_argument('--resamples', type=int, default=RESAMPLES)
    a = ap.parse_args()
    runs = os.path.join(a.root, 'paperc', 'runs'); tag = 'plain' if a.plain else 'final'
    fname = 'per_image_validation.npz' if a.plain else 'per_image_validation_v1_8_final_fixed.npz'
    na, da = load(os.path.join(runs, a.a), fname); nb, db = load(os.path.join(runs, a.b), fname)
    if sorted(na) != sorted(nb): sys.exit('the two runs were evaluated on different images')
    pos = {n: i for i, n in enumerate(nb)}; order = [pos[n] for n in na]
    db = {k: v[order] for k, v in db.items()}; N = len(na)
    one = np.ones((1, N)); pa, pb = metrics(one, da), metrics(one, db)
    W = np.random.default_rng(SEED).multinomial(N, np.full(N, 1.0 / N), size=a.resamples).astype(np.float64)
    ba, bb = metrics(W, da), metrics(W, db)
    print(f'{VERSION} | A = {a.a}, B = {a.b} | {tag} results ({fname}) | {N} validation images, '
          f'{a.resamples} paired resamples (seed {SEED})')
    print(f'\n{"metric":44s} {"A":>7s} {"B":>7s} {"A - B":>8s}   95% interval of A - B   share A <= B')
    res = {'code': VERSION, 'A': a.a, 'B': a.b, 'results': tag, 'file': fname, 'images': N, 'resamples': a.resamples,
           'seed': SEED, 'metrics': {}, 'per_class': {}}
    for k in pa:
        if k.startswith('per_class'): continue
        d = ba[k] - bb[k]; lo, hi = np.nanpercentile(d, [2.5, 97.5]); p = float(np.mean(d <= 0))
        va, vb = float(pa[k][0]), float(pb[k][0])
        flag = '  above 0' if lo > 0 else ('  below 0' if hi < 0 else '')
        print(f'{k:44s} {va:7.4f} {vb:7.4f} {va - vb:+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {p:.3f}{flag}')
        res['metrics'][k] = {'A': round(va, 4), 'B': round(vb, 4), 'diff': round(va - vb, 4),
                             'ci95': [round(float(lo), 4), round(float(hi), 4)], 'share_A_not_better': round(p, 4)}
    for tagk in ('native', '512x512'):
        k = f'per_class {tagk}'
        if k not in pa: continue
        print(f'\nper class, IoU {tagk}: A | B | A - B | 95% interval')
        res['per_class'][tagk] = {}
        for c in range(len(CLASSES)):
            va, vb = pa[k][0, c], pb[k][0, c]; d = ba[k][:, c] - bb[k][:, c]
            if np.isnan(va) or np.isnan(vb): continue
            lo, hi = np.nanpercentile(d, [2.5, 97.5]); flag = '  above 0' if lo > 0 else ('  below 0' if hi < 0 else '')
            print(f'  {CLASSES[c]:15s} {va:.4f} | {vb:.4f} | {va - vb:+.4f} | [{lo:+.4f}, {hi:+.4f}]{flag}'
                  + ('  (T)' if c in TIDX else ''))
            res['per_class'][tagk][CLASSES[c]] = {'A': round(float(va), 4), 'B': round(float(vb), 4),
                                                  'diff': round(float(va - vb), 4),
                                                  'ci95': [round(float(lo), 4), round(float(hi), 4)]}
    print('\nS1 rule: a difference counts as supported only if its whole 95% interval is above 0 ("above 0").')
    out = os.path.join(a.root, 'paperc', f'bootstrap_{a.a}_vs_{a.b}_{tag}_v1_0.json')
    with open(out + '.part', 'w') as f: json.dump(res, f, indent=1)
    os.replace(out + '.part', out); print(f'written {out}')


if __name__ == '__main__':
    main()
