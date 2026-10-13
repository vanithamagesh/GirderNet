"""Paper C (bridge damage), step S0 (v1.3): data check and damage-component relation statistics of dacl10k.

dacl10k: Flotzinger et al., WACV 2024; 9,920 bridge inspection images, 19 classes (13 damage, 6 components), multi-label
polygons (labelme format). Data licence CC BY-NC 4.0. Annotations are public for train (6,935) and validation (975);
testdev and testchallenge images have no public annotations.

No model is trained here. The script
  1. uses a dacl10k zip already on Drive (any *dacl*.zip at the top of MyDrive or in <root>/dacl10k/, or --zip), else
     downloads dacl10k_v2_devphase.zip (5.1 GB, AWS link of the dacl10k toolkit) and keeps a copy on Drive; the sha256
     is compared with the published one,
  2. reads every annotation and reports classes, shape types, image sizes, polygon areas and widths,
  3. rasterizes the polygons (long side --long-side px) and computes, per split, the image co-occurrence (lift),
     the pixel containment and the spatial adjacency of every class pair, each against a chance baseline,
  4. applies the decision rule below, which was fixed before the first run.

Relation measures (pair a -> b, over the images that contain both classes):
  lift          P(a and b in an image) / (P(a) P(b)) over all images of the split
  containment   observed share of a-pixels inside b / share expected if a-pixels fell at random in the image
  adjacency     same with b dilated by a square of half-width --radius px (at --long-side), i.e. a on or next to b
Decision rule (fixed in advance):
  strong pair   >= 20 training images with both classes, and lift >= 2 or adjacency ratio >= 3
  rare class    lower third of the 19 classes by number of training images
  stability     Spearman correlation of log lift, train vs validation, over pairs with >= 10 images in both splits
  GO            >= 10 strong pairs, >= half of the rare classes have a strong partner, stability >= 0.7
  NO-GO         otherwise: the relation idea is too weak and is revised before any training
Relation matrices used later to initialise the model come from the training split only.

Output in <root>/dacl10k/: s0_stats.json, s0_relations_train.npz, s0_relations_validation.npz, fig_s0_relations.png.
Nothing is deleted. Colab (CPU), after mounting Drive:
  !python /content/drive/MyDrive/PaperC_SHM_study/data_check_s0.py --root /content/drive/MyDrive/PaperC_SHM_study
A zip with another name or place: --zip /content/drive/MyDrive/<name>.zip; an unpacked folder: --data <folder>;
force the AWS download: --zip none.
"""
import argparse, glob, hashlib, json, os, shutil, sys, time, urllib.request, zipfile
import numpy as np

CLASSES = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity', 'Spalling',
           'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 'Bearing', 'EJoint', 'Drainage', 'PEquipment',
           'JTape', 'WConccor']                                   # order of the dacl10k toolkit (TARGET_LIST)
URL = 'https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip'
SHA256 = 'dcbcd5fb82699076a2c7f3a72492a9ef798870e0ca1f0c9399360f273ea95260'
SIZE = 5109737241                                               # bytes, from the AWS server
SPLITS = ('train', 'validation')

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study')
ap.add_argument('--work', default='/content/dacl10k', help='local disk for the unpacked data (fast)')
ap.add_argument('--data', default=None, help='unpacked dacl10k folder (contains annotations/ and images/)')
ap.add_argument('--long-side', type=int, default=768, help='rasterization size of the longer image side')
ap.add_argument('--radius', type=int, default=8, help='half-width of the adjacency square in px at --long-side')
ap.add_argument('--max-images', type=int, default=0, help='0 = all; >0 only for a quick trial')
ap.add_argument('--no-drive-copy', action='store_true', help='do not keep a copy of the zip on Drive')
ap.add_argument('--zip', default=None, help='dacl10k zip on Drive; default: search; "none" = download from AWS')
ap.add_argument('--mydrive', default='/content/drive/MyDrive', help='top of the mounted Drive (searched for *dacl*.zip)')
a = ap.parse_args()
OUT = os.path.join(a.root, 'dacl10k'); os.makedirs(OUT, exist_ok=True)
C = len(CLASSES); CI = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------------------------------- data
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def get_data():
    """returns the unpacked data folder. Order: --data folder; a dacl10k zip already on Drive (--zip, or found in
    <root>/dacl10k/ or at the top of MyDrive); otherwise the download from AWS. Each source has its own local copy."""
    if a.data: return a.data
    os.makedirs(a.work, exist_ok=True)
    if a.zip and a.zip.lower() == 'none': found = []
    elif a.zip:
        if not os.path.isfile(a.zip): sys.exit(f'--zip: no file {a.zip}')
        found = [a.zip]
    else:
        found = sorted(set(p for d in (OUT, a.mydrive) for p in glob.glob(os.path.join(d, '*.zip'))
                           if 'dacl' in os.path.basename(p).lower() and 'testchallenge' not in p.lower()))
    src = max(found, key=lambda q: (os.path.getsize(q) == SIZE, os.path.getsize(q))) if found else None  # published zip first
    zlocal = os.path.join(a.work, 'drive_' + os.path.basename(src) if src else 'aws_' + os.path.basename(URL))
    if not os.path.exists(zlocal):
        if src:
            print(f'zip found on Drive: {src} ({os.path.getsize(src) / 1e9:.2f} GB); copying to the local disk ...', flush=True)
            shutil.copy(src, zlocal + '.part'); os.replace(zlocal + '.part', zlocal)
        else:
            req = urllib.request.urlopen(URL); size = int(req.headers.get('Content-Length', 0))
            print(f'no dacl10k zip on Drive; downloading {URL} ({size / 1e9:.2f} GB) ...', flush=True)
            t, done = time.time(), 0
            with open(zlocal + '.part', 'wb') as f:
                while True:
                    b = req.read(1 << 24)
                    if not b: break
                    f.write(b); done += len(b)
                    if done % (1 << 28) < (1 << 24): print(f'  {done / 1e9:.2f} GB, {time.time() - t:.0f} s', flush=True)
            os.replace(zlocal + '.part', zlocal)
    digest = sha256(zlocal); same = digest == SHA256
    print('sha256:', 'identical to the published dev-phase zip' if same else
          f'DIFFERENT from the published dev-phase zip ({digest}); continuing, the folder layout is checked below')
    zdrive = os.path.join(OUT, os.path.basename(URL))
    if src is None and same and not a.no_drive_copy and not (os.path.exists(zdrive) and os.path.getsize(zdrive) == SIZE):
        print('keeping a copy of the zip on Drive for later steps ...', flush=True)
        try: shutil.copy(zlocal, zdrive + '.part'); os.replace(zdrive + '.part', zdrive)
        except OSError as e: print('  copy to Drive failed (Drive full?), continuing with the local zip only:', e)
    dest = os.path.splitext(zlocal)[0] + '_unpacked'
    if not os.path.isdir(dest):
        zipfile.ZipFile(zlocal).extractall(dest + '.part'); os.replace(dest + '.part', dest)
    if find_split_dir(dest, 'annotations', 'train') is None:          # a zip inside the zip (browser download)
        for inner in glob.glob(os.path.join(dest, '**', '*.zip'), recursive=True):
            print('unpacking the inner zip', os.path.relpath(inner, dest), flush=True)
            zipfile.ZipFile(inner).extractall(os.path.splitext(inner)[0])
    if find_split_dir(dest, 'annotations', 'train') is None:
        print('\nthe zip has no annotations/train folder; its top levels are:')
        for dp, dn, fn in os.walk(dest):
            if dp[len(dest):].count(os.sep) <= 2: print(' ', os.path.relpath(dp, dest), f'({len(dn)} folders, {len(fn)} files)')
        sys.exit('not the original dacl10k layout (labelme JSON); run with --zip none to download the original from AWS')
    return dest


def find_split_dir(root, kind, split):
    """annotations/<split> or images/<split>, at any depth below root (the zip may add a top folder)."""
    hits = [p for p in glob.glob(os.path.join(root, '**', kind, split), recursive=True) if os.path.isdir(p)]
    return sorted(hits, key=len)[0] if hits else None


# ---------------------------------------------------------------------------------------------------- geometry
def poly_area_perimeter(p):
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    per = np.sqrt((np.diff(np.vstack([p, p[:1]]), axis=0) ** 2).sum(1)).sum()
    return area, per


def rasterize(ann, L):
    """(C, h, w) bool masks of one annotation at long side L; polygons and rectangles are filled."""
    from PIL import Image, ImageDraw
    W, H = ann['imageWidth'], ann['imageHeight']; s = L / max(W, H)
    w, h = max(1, round(W * s)), max(1, round(H * s))
    masks = np.zeros((C, h, w), bool); layers = {}
    for sh in ann['shapes']:
        lab = sh['label']
        if lab not in CI: continue
        p = np.asarray(sh['points'], float) * s
        if sh.get('shape_type', 'polygon') == 'rectangle' and len(p) == 2:
            (x0, y0), (x1, y1) = p
            p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if len(p) < 3: continue
        img = layers.get(lab)
        if img is None: img = layers[lab] = Image.new('1', (w, h), 0)
        ImageDraw.Draw(img).polygon([tuple(q) for q in p], fill=1, outline=1)
    for lab, img in layers.items(): masks[CI[lab]] = np.asarray(img, bool)
    return masks


def dilate(m, r):
    """dilation with a (2r+1) x (2r+1) square, separable and fast."""
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(maximum_filter1d(m.view(np.uint8), 2 * r + 1, axis=0), 2 * r + 1, axis=1).astype(bool)


# ---------------------------------------------------------------------------------------------------- statistics
def split_stats(files):
    st = {'n_images': 0, 'n_empty': 0, 'sizes': {}, 'shape_types': {}, 'unknown_labels': {}, 'versions': {},
          'img_count': np.zeros(C, int), 'poly_count': np.zeros(C, int), 'area_share': [[] for _ in range(C)],
          'poly_width_rel': [[] for _ in range(C)], 'multi_pixel_share': [], 'own_overlap': np.zeros((C, 2))}
    co = np.zeros((C, C), int)                          # images with both classes
    cont_obs, cont_exp = np.zeros((C, C)), np.zeros((C, C))
    adj_obs, adj_exp = np.zeros((C, C)), np.zeros((C, C))
    for n, f in enumerate(files):
        ann = json.load(open(f)); st['n_images'] += 1
        W, H = ann['imageWidth'], ann['imageHeight']; key = f'{W}x{H}'
        st['sizes'][key] = st['sizes'].get(key, 0) + 1
        v = str(ann.get('dacl10k_version')); st['versions'][v] = st['versions'].get(v, 0) + 1
        present = np.zeros(C, bool)
        for sh in ann['shapes']:
            t = sh.get('shape_type', 'polygon'); st['shape_types'][t] = st['shape_types'].get(t, 0) + 1
            lab = sh['label']
            if lab not in CI: st['unknown_labels'][lab] = st['unknown_labels'].get(lab, 0) + 1; continue
            i = CI[lab]; present[i] = True; st['poly_count'][i] += 1
            p = np.asarray(sh['points'], float)
            if len(p) >= 3:
                ar, pe = poly_area_perimeter(p)
                if pe > 0: st['poly_width_rel'][i].append(2 * ar / pe / max(W, H))   # mean width / long side
        if not present.any(): st['n_empty'] += 1; continue
        st['img_count'] += present; co += np.outer(present, present)
        m = rasterize(ann, a.long_side); N = m[0].size
        area = m.reshape(C, -1).sum(1)
        for i in np.flatnonzero(present): st['area_share'][i].append(area[i] / N)
        lab_px = m.any(0); cnt = m.sum(0)
        if lab_px.any(): st['multi_pixel_share'].append(float((cnt >= 2).sum() / lab_px.sum()))
        idx = np.flatnonzero(present & (area > 0))
        dil = {j: dilate(m[j], a.radius) for j in idx}
        for i in idx:
            others = cnt[m[i]] >= 2; st['own_overlap'][i] += (others.sum(), m[i].sum())
            for j in idx:
                if i == j: continue
                cont_obs[i, j] += (m[i] & m[j]).sum(); cont_exp[i, j] += area[i] * area[j] / N
                adj_obs[i, j] += (m[i] & dil[j]).sum(); adj_exp[i, j] += area[i] * dil[j].sum() / N
        if (n + 1) % 1000 == 0: print(f'  {n + 1} / {len(files)} images', flush=True)
    n_img = st['n_images']
    P = st['img_count'] / n_img
    with np.errstate(divide='ignore', invalid='ignore'):
        lift = (co / n_img) / np.outer(P, P)
        cond = co / st['img_count'][:, None]           # P(b | a), row a
        cont_ratio, adj_ratio = cont_obs / cont_exp, adj_obs / adj_exp
    rel = {'cooc': co, 'lift': lift, 'p_b_given_a': cond, 'containment_ratio': cont_ratio, 'adjacency_ratio': adj_ratio,
           'containment_obs_px': cont_obs, 'adjacency_obs_px': adj_obs, 'img_count': st['img_count']}
    return st, rel


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y); return float(r.correlation if hasattr(r, 'correlation') else r[0])


def summary_class_table(st):
    rows = []
    for i, c in enumerate(CLASSES):
        ws = np.asarray(st['poly_width_rel'][i]); ar = np.asarray(st['area_share'][i])
        ov = st['own_overlap'][i]
        rows.append({'class': c, 'images': int(st['img_count'][i]), 'polygons': int(st['poly_count'][i]),
                     'median_area_share_pct': round(100 * float(np.median(ar)), 3) if ar.size else None,
                     'median_width_rel_pct': round(100 * float(np.median(ws)), 3) if ws.size else None,
                     'thin_share_pct': round(100 * float((ws < 0.01).mean()), 1) if ws.size else None,
                     'overlap_with_other_class_pct': round(100 * ov[0] / ov[1], 1) if ov[1] else None})
    return rows


def plot(rel, ok, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', ['#2b6cb0', '#e2e2e2', '#c0392b']); cmap.set_bad('white')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7), constrained_layout=True)
    for ax, key, title in ((axs[0], 'lift', '(a) Image co-occurrence: log$_2$ lift'),
                           (axs[1], 'adjacency_ratio', '(b) Spatial adjacency: log$_2$ (observed / chance)')):
        with np.errstate(divide='ignore', invalid='ignore'): z = np.log2(rel[key])
        z = np.where(ok & np.isfinite(z), z, np.nan); np.fill_diagonal(z, np.nan)
        im = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=-3, vmax=3, interpolation='nearest')
        ax.set_xticks(range(C)); ax.set_yticks(range(C))
        ax.set_xticklabels(CLASSES, rotation=90); ax.set_yticklabels(CLASSES)
        ax.set_title(title, fontsize=8, loc='left'); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[-3, -2, -1, 0, 1, 2, 3]); cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
    for ax in axs: ax.set_xlabel('class b')
    axs[0].set_ylabel('class a')            # blank cells: fewer than 20 training images with both classes
    fig.savefig(path, dpi=300); plt.close(fig)


# ---------------------------------------------------------------------------------------------------- main
root = get_data()
S = {'version': 'S0 v1.3', 'classes': CLASSES, 'long_side': a.long_side, 'radius': a.radius, 'splits': {}}
RELS = {}
for sp in SPLITS:
    adir, idir = find_split_dir(root, 'annotations', sp), find_split_dir(root, 'images', sp)
    files = sorted(glob.glob(os.path.join(adir, '*.json'))) if adir else []
    if a.max_images: files = files[:a.max_images]
    n_img = len(os.listdir(idir)) if idir else 0
    print(f'\n=== {sp}: {len(files)} annotation files, {n_img} images ({adir})', flush=True)
    if not files: print('  no annotations found'); continue
    st, rel = split_stats(files); RELS[sp] = rel
    rows = summary_class_table(st)
    print(f'images {st["n_images"]}, without any label {st["n_empty"]}; shape types {st["shape_types"]}; '
          f'unknown labels {st["unknown_labels"]}; versions {st["versions"]}')
    sizes = sorted(st['sizes'].items(), key=lambda kv: -kv[1])
    print('most frequent image sizes:', sizes[:6], f'({len(sizes)} different)')
    mp = np.asarray(st['multi_pixel_share'])
    print(f'labelled pixels with >= 2 classes: mean over images {100 * mp.mean():.1f}%')
    print(f'\n{"class":15s}{"images":>8s}{"polygons":>10s}{"med.area%":>11s}{"med.width%":>12s}{"thin%":>8s}{"overlap%":>10s}')
    for r in rows:
        print(f'{r["class"]:15s}{r["images"]:8d}{r["polygons"]:10d}{str(r["median_area_share_pct"]):>11s}'
              f'{str(r["median_width_rel_pct"]):>12s}{str(r["thin_share_pct"]):>8s}{str(r["overlap_with_other_class_pct"]):>10s}')
    S['splits'][sp] = {'n_images': st['n_images'], 'n_images_folder': n_img, 'n_empty': st['n_empty'],
                       'shape_types': st['shape_types'], 'unknown_labels': st['unknown_labels'],
                       'versions': st['versions'], 'n_sizes': len(sizes), 'top_sizes': sizes[:10],
                       'multi_label_pixel_share_pct': round(100 * float(mp.mean()), 2), 'classes': rows}
    np.savez(os.path.join(OUT, f's0_relations_{sp}.npz'), classes=np.array(CLASSES),
             **{k: np.asarray(v, float) for k, v in rel.items()})

# ---------------------------------------------------------------------------------------------------- decision
if 'train' in RELS:
    R = RELS['train']; co = R['cooc']; ok = co >= 20; np.fill_diagonal(ok, False)
    strong = ok & ((R['lift'] >= 2) | (R['adjacency_ratio'] >= 3))
    pairs = sorted([(CLASSES[i], CLASSES[j], int(co[i, j]), round(float(R['lift'][i, j]), 2),
                     round(float(R['adjacency_ratio'][i, j]), 2), round(float(R['containment_ratio'][i, j]), 2),
                     round(float(R['p_b_given_a'][i, j]), 3))
                    for i, j in zip(*np.nonzero(strong))], key=lambda t: -t[4])
    n_strong_unordered = int(np.triu(strong | strong.T, 1).sum())
    order = np.argsort(R['img_count']); rare = [CLASSES[i] for i in order[:C // 3]]
    rare_ok = [c for c in rare if (strong[CI[c]] | strong[:, CI[c]]).any()]
    stab = None
    if 'validation' in RELS:
        V = RELS['validation']; both = (co >= 10) & (V['cooc'] >= 10); np.fill_diagonal(both, False)
        both = np.triu(both, 1)
        if both.sum() >= 5: stab = spearman(np.log2(R['lift'][both]), np.log2(V['lift'][both]))
    go = n_strong_unordered >= 10 and len(rare_ok) >= len(rare) / 2 and stab is not None and stab >= 0.7
    print('\n=== relation strength (training split; rule fixed in advance)')
    print(f'strong class pairs (unordered): {n_strong_unordered}  [need >= 10]')
    print(f'rare classes: {rare}; with a strong partner: {rare_ok}  [need >= {len(rare) / 2:.1f}]')
    print(f'stability, Spearman of log lift train vs validation: {stab if stab is None else round(stab, 3)}  [need >= 0.7]')
    print('DECISION:', 'GO - build the relation model' if go else 'NO-GO - revise the relation idea before training')
    print(f'\n{"a":15s}{"b":15s}{"images":>8s}{"lift":>7s}{"adj":>7s}{"cont":>7s}{"P(b|a)":>8s}   (strong pairs, by adjacency)')
    for p in pairs[:40]: print(f'{p[0]:15s}{p[1]:15s}{p[2]:8d}{p[3]:7.2f}{p[4]:7.2f}{p[5]:7.2f}{p[6]:8.3f}')
    S['decision'] = {'n_strong_pairs_unordered': n_strong_unordered, 'rare_classes': rare, 'rare_with_partner': rare_ok,
                     'stability_spearman': stab, 'go': bool(go), 'strong_pairs': pairs}
    plot(R, ok, os.path.join(OUT, 'fig_s0_relations.png'))
json.dump(S, open(os.path.join(OUT, 's0_stats.json'), 'w'), indent=1)
print('\nwritten', os.path.join(OUT, 's0_stats.json'), 'and the relation files in', OUT)
