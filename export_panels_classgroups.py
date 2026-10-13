"""Paper C: raw panels for the additional segmentation figures, small enough to upload (about 10 MB).
Version 1.1, 12 October 2026 (from v1.0: images chosen for four more class groups; zero-shot examples on S2DS and
DeepCrack). The 600-dpi figures are drawn from these files outside Colab.

Part 1, dacl10k validation (no model is run; the stored predictions are used). For each class group, the first 4
validation images by name that meet the group's rule, leaving out images already shown (val 0048, 0049, 0054, 0056)
and images already taken by an earlier group:
  surface   Weathering with Wetspot and Efflorescence (if fewer than 4: Weathering with Wetspot or Efflorescence)
  cracks    ACrack with Crack (if fewer than 4: ACrack)
  objects   at least two of Bearing, Drainage, EJoint, JTape, PEquipment
  weak      Rockpocket or WConccor
For each image: <stem>_1024.jpg (JPEG quality 95, no chroma subsampling) and <stem>_masks.npz (19-class masks at long
side 1024, bits packed along the width: labels, B1, baseline, ours, stage2).
Part 2, zero-shot (both large models run once per image at the input size each chose; GPU or CPU, a few minutes on
CPU): S2DS, the first 4 test patches by name whose labels contain at least two of crack, spalling, corrosion and
efflorescence; DeepCrack, 4 test images evenly spaced by name. Saved: the image as zs_<dataset>_<name>.jpg, the label
and the mapped predictions (threshold 0.5, no TTA) as zs_<dataset>_<name>.npz. Skip with --no-zeroshot.
Writes <root>/paperc/panels_v1_1/ and <root>/paperc/panels_v1_1.zip with manifest.json (rules, counts, chosen images).
Needs pipeline.py, figs_helpers.py and external_eval.py in the same folder (all already on Drive).
Colab (CPU runtime is enough):
  from google.colab import drive; drive.mount('/content/drive')
  !python export_panels_classgroups.py
"""
import argparse, importlib.util, io, json, os, shutil, sys, time
import numpy as np

VERSION = 'export_panels_classgroups.py'
KEY = {'B1-s0': 'B1', 'B3-L-s0': 'baseline', 'B3-ovl-L0.1-s0': 'ours', 'Full-on-B3-ovl-L0.1-s0-skelT-s0': 'stage2'}
SHOWN = {'dacl10k_v2_validation_0048.jpg', 'dacl10k_v2_validation_0049.jpg', 'dacl10k_v2_validation_0054.jpg',
         'dacl10k_v2_validation_0056.jpg'}
OBJECTS = ('Bearing', 'Drainage', 'EJoint', 'JTape', 'PEquipment')
GROUPS = [('surface', [lambda h: h('Weathering') and h('Wetspot') and h('Efflorescence'),
                       lambda h: h('Weathering') and (h('Wetspot') or h('Efflorescence'))]),
          ('cracks', [lambda h: h('ACrack') and h('Crack'), lambda h: h('ACrack')]),
          ('objects', [lambda h: sum(h(c) for c in OBJECTS) >= 2]),
          ('weak', [lambda h: h('Rockpocket') or h('WConccor')])]
ZS_RUNS = ['B3-L-s0', 'B3-ovl-L0.1-s0']


def log(*a): print(*a, flush=True)


def load(name, path):
    if not os.path.exists(path): sys.exit(f'{os.path.basename(path)} not found next to {VERSION}')
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def save_jpg(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path, quality=95, subsampling=0)


def part1(a, PC, FG, out, man):
    idx = json.load(open(os.path.join(a.root, 'paperc', 's2_index.json')))['index']
    val = sorted([m for m in idx if m['split'] == 'validation'], key=lambda m: m['name'])
    taken = set(SHOWN); picks = []
    for group, rules in GROUPS:
        chosen = []; counts = []
        for rule in rules:
            ok = [m for m in val if rule(lambda c, m=m: bool(m['presence'][PC.CI[c]]))]
            counts.append(len(ok))
            for m in ok:
                if len(chosen) < a.n and m['name'] not in taken: chosen.append(m); taken.add(m['name'])
        man['groups'][group] = {'qualifying_per_rule': counts, 'images': [m['name'] for m in chosen]}
        log(f'  {group}: {counts} images qualify (per rule); chosen: ' + ', '.join(m['name'][-8:-4] for m in chosen))
        picks += chosen
    srcs = {}
    for _, run, kind in FG.MODELS:
        src = FG.pred_source(os.path.join(a.runs, run), kind)
        if src: srcs[run] = src
        else: log(f'  no saved predictions for {run}: left out')
    man['models'] = {KEY[r]: {'run': r, 'tag': s[2], 'mask_key': s[1]} for r, s in srcs.items()}
    imgs = FG.Images(a)
    for m in picks:
        W, H = m['W'], m['H']; s = PC.RULES['global_long_side'] / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
        stem = os.path.splitext(m['name'])[0]
        save_jpg(FG.load_rgb(imgs.read(m['name']), W, H, w, h), os.path.join(out, f'{stem}_1024.jpg'))
        gt = PC.rasterize_window(PC.polygons(json.loads(imgs.read(stem + '.json'))), s, 0, 0, w, h)
        arrays = {'labels': np.packbits(np.asarray(gt, bool), axis=-1)}
        for run, src in srcs.items():
            p = FG.load_mask(src, m['name'])
            if p is None or p.shape != gt.shape: log(f'  {m["name"]}: {run} prediction missing or of another size'); continue
            arrays[KEY[run]] = np.packbits(p, axis=-1)
        np.savez_compressed(os.path.join(out, f'{stem}_masks.npz'), shape=np.array(gt.shape), **arrays)
        man['images'].append({'name': m['name'], 'W': W, 'H': H, 'masks': sorted(arrays)})


def part2(a, PC, XD, out, man):
    XD.PC = PC; PC.ensure_smp()
    torch, _, _ = PC.torch_mods(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from PIL import Image
    data = {}
    _, te = XD.get_s2ds(a)
    good = []
    for it in sorted(te, key=lambda t: t['name']):
        lab = XD.s2ds_label(it['label'])
        if sum(bool((lab == XD.S2DS_ID_OF[c]).any()) for c in XD.S2DS_MAP) >= 2: good.append(it)
    man['zeroshot']['s2ds_qualifying'] = len(good); data['s2ds'] = good[:a.n]
    _, te = XD.get_deepcrack(a); te = sorted(te, key=lambda t: t['name'])
    data['deepcrack'] = [te[int(round(i))] for i in np.linspace(0, len(te) - 1, a.n)] if te else []
    chosenL = {}
    for ds in data:
        for run in ZS_RUNS:
            f = os.path.join(a.root, 'paperc', 'xdata_v1_0', f'{ds}_{run}.json')
            chosenL[(ds, run)] = json.load(open(f))['chosen_L'] if os.path.exists(f) else 1024
    store = {(ds, it['name']): {} for ds in data for it in data[ds]}
    for run in ZS_RUNS:
        rdir = os.path.join(a.runs, run)
        meta = json.load(open(os.path.join(rdir, 'config.json'))); cfg = dict(PC.CONFIGS[meta['model']])
        model = PC.build_model(cfg, pretrained=False, small=a.small).to(dev).eval()
        best = torch.load(os.path.join(rdir, 'best.pt'), map_location='cpu')
        model.load_state_dict({k: v.float() for k, v in best['model'].items()}); del best
        for ds, items in data.items():
            mp = XD.S2DS_MAP if ds == 's2ds' else XD.DC_MAP; L = chosenL[(ds, run)]
            for it in items:
                t0 = time.time(); p = XD.predict(model, cfg, dev, it['image'], L) > 0.5
                for c, src in mp.items():
                    pr = np.zeros(p.shape[1:], bool)
                    for k in src: pr |= p[PC.CI[k]]
                    store[(ds, it['name'])][f'{KEY[run]}_{c}'] = np.packbits(pr, axis=-1)
                log(f'  {run} | {ds} {it["name"]} at L {L}: {time.time() - t0:.1f} s')
        del model
        if dev.type == 'cuda': torch.cuda.empty_cache()
    for ds, items in data.items():
        for it in items:
            img = np.asarray(Image.open(it['image']).convert('RGB'))
            lab = XD.s2ds_label(it['label']) if ds == 's2ds' else (np.asarray(Image.open(it['label']).convert('L')) > 127).astype(np.uint8)
            save_jpg(img, os.path.join(out, f'zs_{ds}_{it["name"]}.jpg'))
            np.savez_compressed(os.path.join(out, f'zs_{ds}_{it["name"]}.npz'), label=lab, shape=np.array(lab.shape),
                                **store[(ds, it['name'])])
            man['zeroshot'].setdefault(ds, []).append({'name': it['name'], 'size': list(lab.shape)})
    man['zeroshot']['input_long_side'] = {f'{ds}/{run}': L for (ds, run), L in chosenL.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--zip', default=None); ap.add_argument('--data', default=None)
    ap.add_argument('--work', default='/content/xdata')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--no-zeroshot', action='store_true')
    ap.add_argument('--small', action='store_true', help='test only: tiny models')
    a = ap.parse_args(); a.runs = os.path.join(a.root, 'paperc', 'runs')
    here = os.path.dirname(os.path.abspath(__file__))
    PC = load('paperc', os.path.join(here, 'pipeline.py')); FG = load('paperc_figs', os.path.join(here, 'figs_helpers.py'))
    FG.PC = PC
    out = os.path.join(a.root, 'paperc', 'panels_v1_1'); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    man = {'code': VERSION, 'classes': list(PC.CLASSES), 'groups': {}, 'images': [], 'zeroshot': {}}
    log(f'{VERSION} | part 1: dacl10k validation images for {len(GROUPS)} class groups')
    part1(a, PC, FG, out, man)
    if not a.no_zeroshot:
        log('part 2: zero-shot examples (S2DS, DeepCrack)')
        try: part2(a, PC, load('paperc_xdata', os.path.join(here, 'external_eval.py')), out, man)
        except Exception as e:
            import traceback; traceback.print_exc(); man['zeroshot']['error'] = repr(e)
            log('part 2 FAILED (part 1 is complete and is written anyway)')
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    z = shutil.make_archive(out, 'zip', out)
    log(f'\nwritten {z}: {os.path.getsize(z) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
