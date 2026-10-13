"""Paper C: zero-shot transfer of the trained dacl10k models to other datasets (no training). Version 1.2, 12 October 2026 (v1.1: installs segmentation-models-pytorch if missing; v1.2: same S2DS crack scores about
20x faster: the lines are thinned once per image and the tolerances use one distance transform instead of repeated
dilations; checked identical to v1.1 and to the official OpenCV code).

Datasets (test splits):
  S2DS (Benz & Rodehorst 2022): 93 patches of 1024 x 1024; classes crack, spalling, corrosion, efflorescence (and
       vegetation, control point, not in dacl10k). Download: Google Drive file 1PQ50QKfy2vnDOHSmw5bpBFi33hZsSXuM
       (academic use only), or put the zip at <root>/xdata/s2ds.zip.
  DeepCrack (Liu et al. 2019): 237 images of 544 x 384, binary crack labels. Download: github.com/yhlleo/DeepCrack.
Models: the best epoch of each run given (default: our model B3-ovl-L0.1-s0 and the baseline B3-L-s0). No TTA,
threshold 0.5, one prediction per image.

Fixed before running:
  class mapping: S2DS crack <- dacl10k Crack or ACrack; spalling <- Spalling; corrosion <- Rust; efflorescence <-
  Efflorescence. DeepCrack crack <- Crack or ACrack.
  input size: the model sees the image at long side L in (512, 768, 1024). L is chosen per model and dataset on the
  dataset's own non-test images (S2DS validation split; every third DeepCrack training image), by the mean IoU of the
  mapped classes (S2DS crack: line-based tolerant IoU at 4 px, as its labels are lines). The test split is then scored
  once with that L; the other two L are written as supplementary results.
Scoring (whole split, at the label resolution of the dataset):
  S2DS, as in the S2DS evaluate.py: areal classes with a 2-pixel ignore band at their edges; crack with the
  line-based tolerant IoU (ltIoU) at tolerances 0, 2, 4, 8, 16, 32 px; plus plain crack IoU and clDice.
  DeepCrack: crack IoU, precision, recall, F1 and clDice.
Runs on a T4 (32-bit there, as the T4 has no bfloat16). CPU works too, slowly.

  from google.colab import drive; drive.mount('/content/drive')
  !python external_eval.py
Writes <root>/paperc/xdata_v1_0/<dataset>_<run>.json and prints a summary table.
"""
import argparse, glob, importlib.util, json, os, re, sys, time, urllib.request, zipfile
import numpy as np

VERSION = 'external_eval.py'
S2DS_ID = '1PQ50QKfy2vnDOHSmw5bpBFi33hZsSXuM'
DEEPCRACK_URL = 'https://github.com/yhlleo/DeepCrack/raw/master/dataset/DeepCrack.zip'
LONG_SIDES = (512, 768, 1024)
TOLS = (0, 2, 4, 8, 16, 32)
S2DS_MAP = {'crack': ('Crack', 'ACrack'), 'spalling': ('Spalling',), 'corrosion': ('Rust',),
            'efflorescence': ('Efflorescence',)}
S2DS_ID_OF = {'crack': 1, 'spalling': 2, 'corrosion': 3, 'efflorescence': 4}         # S2DS class indices
DC_MAP = {'crack': ('Crack', 'ACrack')}
PC = None                                                                            # paperc_v1_8 module


def log(*a): print(*a, flush=True)


def load_paperc(path):
    global PC
    spec = importlib.util.spec_from_file_location('paperc', path); PC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PC)


# ------------------------------------------------------------------------------------------------ data
def get_s2ds(a):
    d = os.path.join(a.work, 's2ds')
    if not glob.glob(os.path.join(d, '**', '[0-9][0-9][0-9]_lab.png'), recursive=True):
        z = os.path.join(a.root, 'xdata', 's2ds.zip')
        if not os.path.exists(z):
            z = os.path.join(a.work, 's2ds.zip')
            if not os.path.exists(z):
                log('downloading S2DS from Google Drive ...')
                try: import gdown
                except ImportError:
                    import subprocess; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown']); import gdown
                os.makedirs(a.work, exist_ok=True); gdown.download(id=S2DS_ID, output=z, quiet=False)
        zipfile.ZipFile(z).extractall(d)
    labs = sorted(glob.glob(os.path.join(d, '**', '[0-9][0-9][0-9]_lab.png'), recursive=True))
    items = {'train': [], 'val': [], 'test': []}
    for lp in labs:
        n = int(os.path.basename(lp)[:3]); folder = os.path.basename(os.path.dirname(lp)).lower()
        img = next((p for p in (lp.replace('_lab.png', e) for e in ('.png', '.jpg', '.JPG')) if os.path.exists(p)), None)
        if img is None: continue
        split = ('test' if 'test' in folder else 'val' if folder.startswith('val') else 'train' if 'train' in folder
                 else ('test' if n >= 650 else 'val' if n >= 563 else 'train'))   # S2DS README: 563 / 87 / 93
        items[split].append({'name': f's2ds_{n:03d}', 'image': img, 'label': lp})
    log(f'S2DS: train {len(items["train"])}, validation {len(items["val"])}, test {len(items["test"])} images '
        '(README: 563 / 87 / 93)')
    return items['val'], items['test']


def get_deepcrack(a):
    d = os.path.join(a.work, 'deepcrack')
    if not os.path.isdir(os.path.join(d, 'test_lab')):
        z = os.path.join(a.work, 'DeepCrack.zip')
        if not os.path.exists(z):
            log('downloading DeepCrack from GitHub ...'); os.makedirs(a.work, exist_ok=True)
            urllib.request.urlretrieve(DEEPCRACK_URL, z + '.part'); os.replace(z + '.part', z)
        zipfile.ZipFile(z).extractall(d)

    def items(split):
        out = []
        for lp in sorted(glob.glob(os.path.join(d, f'{split}_lab', '*.png'))):
            stem = os.path.splitext(os.path.basename(lp))[0]
            img = next((p for p in glob.glob(os.path.join(d, f'{split}_img', stem + '.*'))), None)
            if img: out.append({'name': f'deepcrack_{stem}', 'image': img, 'label': lp})
        return out
    tr, te = items('train'), items('test')
    log(f'DeepCrack: train {len(tr)} (every third used to choose the input size: {len(tr[::3])}), test {len(te)} (paper: 300 / 237)')
    return tr[::3], te


def s2ds_label(path):
    """class index per pixel exactly as S2DS lab2class (RGB): white crack, red spalling, yellow corrosion,
    cyan efflorescence, green vegetation, blue control point, everything else background."""
    from PIL import Image
    lab = np.asarray(Image.open(path).convert('RGB')).copy(); lab[lab < 200] = 0
    out = np.zeros(lab.shape[:2], np.uint8)
    for k, rgb in ((1, (255, 255, 255)), (2, (255, 0, 0)), (3, (255, 255, 0)), (4, (0, 255, 255)), (5, (0, 255, 0)),
                   (6, (0, 0, 255))):
        out[(lab == rgb).all(2)] = k
    return out


def s2ds_ignore(lab, tol=2):
    """S2DS boundary_tolerance: pixels within `tol` of an edge between areal labels (cracks excluded) are ignored.
    cv2.Laplacian (3 x 3, mirrored border) and cv2.dilate with a disk, written with scipy / scikit-image."""
    from scipy.ndimage import convolve, binary_dilation
    from skimage.morphology import disk
    l = np.where(lab == 1, 0, lab).astype(np.float64)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64)
    edge = convolve(l, k, mode='mirror') != 0
    ign = binary_dilation(edge, disk(tol).astype(bool))
    return ign & (lab != 1)


def lines_tolerant_all(pred, gt, tols):
    """S2DS transform2lines for several tolerances at once: thin both masks once; a pixel lies within tol of a line
    exactly when its Euclidean distance to the line is <= tol, which equals dilation with skimage disk(tol). Returns
    {tol: (intersection, union)} of the official line-based tolerant IoU for the crack class."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import thin
    po, to = thin(pred), thin(gt); out = {}
    dp = distance_transform_edt(~po) if po.any() else np.full(po.shape, np.inf)
    dt = distance_transform_edt(~to) if to.any() else np.full(to.shape, np.inf)
    for tol in tols:
        pd_, td = dp <= tol, dt <= tol
        tp = to & pd_; fp = po & ~td; fn = to & ~tp
        o, g = tp | fp, tp | fn
        out[tol] = (int((o & g).sum()), int((o | g).sum()))
    return out


# ------------------------------------------------------------------------------------------------ prediction
def predict(model, cfg, dev, path, L):
    """probabilities of the 19 dacl10k classes at the image's own size, the model seeing the image at long side L."""
    torch, _, F = PC.torch_mods()
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('RGB'); W, H = im.size; s = L / max(W, H)
        img = np.asarray(im.resize((max(16, round(W * s)), max(16, round(H * s))), Image.BILINEAR))
    x, (h, w) = PC.norm_pad(img)
    with torch.no_grad():
        lo = PC.Predictor(cfg, model, dev).net(model, x.to(dev), cfg)[:, :, :h, :w]
        p = torch.sigmoid(F.interpolate(lo, size=(H, W), mode='bilinear', align_corners=False)[0])
    return p.cpu().numpy()


def score(items, model, cfg, dev, L, kind, full):
    """whole-split counts for one input size. kind 's2ds' or 'deepcrack'; full=False: only what selection needs."""
    mp = S2DS_MAP if kind == 's2ds' else DC_MAP
    acc = {c: np.zeros(4, np.int64) for c in mp}                     # inter, union, pred, gt
    lt = {t: np.zeros(2, np.int64) for t in (TOLS if full else (4,))} if kind == 's2ds' else {}
    cl = np.zeros(4, np.int64); per_image = []; t0 = time.time()
    for it in items:
        p = predict(model, cfg, dev, it['image'], L) > 0.5
        if kind == 's2ds':
            lab = s2ds_label(it['label']); valid = ~s2ds_ignore(lab)
            gts = {c: lab == S2DS_ID_OF[c] for c in mp}
        else:
            from PIL import Image
            gts = {'crack': np.asarray(Image.open(it['label']).convert('L')) > 127}; valid = np.ones_like(gts['crack'])
        rec = {}
        for c, src in mp.items():
            pr = np.zeros(p.shape[1:], bool)
            for k in src: pr |= p[PC.CI[k]]
            g = gts[c]; v = valid if (kind == 's2ds' and c != 'crack') else np.ones_like(g)
            r = np.array([(pr & g & v).sum(), ((pr | g) & v).sum(), (pr & v).sum(), (g & v).sum()], np.int64)
            acc[c] += r; rec[c] = r
            if c == 'crack':
                if lt:
                    for t, v in lines_tolerant_all(pr, g, list(lt)).items(): lt[t] += v
                if full:
                    sp, sg = PC.skeleton(pr), PC.skeleton(g)
                    cl += [int((sp & g).sum()), int(sp.sum()), int((sg & pr).sum()), int(sg.sum())]
        per_image.append((it['name'], rec))
    with np.errstate(invalid='ignore', divide='ignore'):
        res = {'L': L, 'images': len(items), 'seconds_per_image': round((time.time() - t0) / max(1, len(items)), 3),
               'IoU': {c: (round(float(v[0] / v[1]), 4) if v[1] else None) for c, v in acc.items()}}
        if lt: res['crack_ltIoU'] = {str(t): (round(float(v[0] / v[1]), 4) if v[1] else None) for t, v in lt.items()}
        if kind == 'deepcrack':
            v = acc['crack']; P, R = v[0] / max(v[2], 1), v[0] / max(v[3], 1)
            res.update({'precision': round(float(P), 4), 'recall': round(float(R), 4),
                        'F1': round(float(2 * P * R / (P + R)), 4) if P + R > 0 else 0.0})
        if full:
            tp, ts = cl[0] / max(cl[1], 1), cl[2] / max(cl[3], 1)
            res['crack_clDice'] = round(float(2 * tp * ts / (tp + ts)), 4) if tp + ts > 0 else 0.0
    sel = [res['crack_ltIoU']['4'] if (kind == 's2ds' and c == 'crack') else res['IoU'][c] for c in mp]
    res['selection_score'] = round(float(np.mean([x for x in sel if x is not None])), 4)
    res['counts'] = {c: v.tolist() for c, v in acc.items()}
    return res, per_image


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['B3-ovl-L0.1-s0', 'B3-L-s0'])
    ap.add_argument('--datasets', nargs='+', default=['s2ds', 'deepcrack'], choices=['s2ds', 'deepcrack'])
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--work', default='/content/xdata')
    ap.add_argument('--paperc', default=None, help='path of pipeline.py (default: next to this script)')
    ap.add_argument('--max-images', type=int, default=0, help='test only')
    ap.add_argument('--small', action='store_true', help='test only: tiny models')
    a = ap.parse_args()
    load_paperc(a.paperc or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline.py'))
    PC.ensure_smp()                                            # v1.1: build_model imports it
    torch, _, _ = PC.torch_mods(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'
    log(f'{VERSION} | {gpu} | runs {a.runs} | datasets {a.datasets} | input long sides {LONG_SIDES}')
    out_dir = os.path.join(a.root, 'paperc', 'xdata_v1_0'); os.makedirs(out_dir, exist_ok=True)
    data = {}
    for ds in a.datasets:
        sel, test = get_s2ds(a) if ds == 's2ds' else get_deepcrack(a)
        if a.max_images: sel, test = sel[:a.max_images], test[:a.max_images]
        if not test: sys.exit(f'{ds}: no test images found')
        data[ds] = (sel, test)
    summary = []
    for run in a.runs:
        rdir = os.path.join(a.root, 'paperc', 'runs', run)
        meta = json.load(open(os.path.join(rdir, 'config.json'))); cfg = dict(PC.CONFIGS[meta['model']])
        if cfg.get('stage') == 2: sys.exit(f'{run}: stage-2 runs are not supported here')
        model = PC.build_model(cfg, pretrained=False, small=a.small).to(dev).eval()
        best = torch.load(os.path.join(rdir, 'best.pt'), map_location='cpu')
        model.load_state_dict({k: v.float() for k, v in best['model'].items()}); del best
        for ds, (sel, test) in data.items():
            path = os.path.join(out_dir, f'{ds}_{run}.json')
            if os.path.exists(path):
                res = json.load(open(path)); log(f'{ds} / {run}: already done ({path})')
            else:
                log(f'{ds} / {run}: choosing the input size on {len(sel)} non-test images ...')
                choice = {L: score(sel, model, cfg, dev, L, ds, False)[0] for L in LONG_SIDES}
                for L, r in choice.items(): log(f'   L {L}: selection score {r["selection_score"]:.4f} ({r["seconds_per_image"]} s/image)')
                Lc = max(LONG_SIDES, key=lambda L: (choice[L]['selection_score'], -abs(L - 1024)))
                log(f'   chosen L = {Lc}; scoring the {len(test)} test images at all three sizes ...')
                tests, per = {}, {}
                for L in LONG_SIDES:
                    tests[L], per[L] = score(test, model, cfg, dev, L, ds, True)
                res = {'code': VERSION, 'dataset': ds, 'run': run, 'gpu': gpu, 'mapping': S2DS_MAP if ds == 's2ds' else DC_MAP,
                       'selection_split': 'S2DS validation' if ds == 's2ds' else 'every third DeepCrack training image',
                       'selection': {str(L): r for L, r in choice.items()}, 'chosen_L': Lc,
                       'test': tests[Lc], 'test_all_L': {str(L): r for L, r in tests.items()},
                       'per_image_test': {str(L): [[n, {c: v.tolist() for c, v in rec.items()}] for n, rec in per[L]]
                                          for L in LONG_SIDES}}
                with open(path + '.part', 'w') as f: json.dump(res, f)
                os.replace(path + '.part', path); log(f'   written {path}')
            summary.append(res)
        del model
        if dev.type == 'cuda': torch.cuda.empty_cache()
    log('\nZERO-SHOT TRANSFER (no training on these datasets; threshold 0.5; no TTA; input size chosen on non-test images)')
    for ds in a.datasets:
        rows = [r for r in summary if r['dataset'] == ds]
        if ds == 's2ds':
            log(f'\nS2DS test ({rows[0]["test"]["images"]} patches): crack ltIoU at 2 / 4 / 8 px | crack IoU | crack clDice | '
                'spalling | corrosion (Rust) | efflorescence | mean of 4 (crack: ltIoU 4 px)')
            for r in rows:
                t = r['test']; lt = t['crack_ltIoU']; I = t['IoU']
                log(f'  {r["run"]:16s} L {r["chosen_L"]:4d}  {lt["2"]} / {lt["4"]} / {lt["8"]} | {I["crack"]} | {t["crack_clDice"]} | '
                    f'{I["spalling"]} | {I["corrosion"]} | {I["efflorescence"]} | {t["selection_score"]}')
        else:
            log(f'\nDeepCrack test ({rows[0]["test"]["images"]} images): crack IoU | precision | recall | F1 | clDice')
            for r in rows:
                t = r['test']
                log(f'  {r["run"]:16s} L {r["chosen_L"]:4d}  {t["IoU"]["crack"]} | {t["precision"]} | {t["recall"]} | {t["F1"]} | {t["crack_clDice"]}')
        log('  (all three input sizes on the test split are in the json files, as supplementary results)')
    log(f'\nwritten {out_dir}')


if __name__ == '__main__':
    main()
