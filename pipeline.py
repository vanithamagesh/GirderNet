"""Paper C (dacl10k v2, bridge damage): shared code and pipeline steps. Version 1.10, 12 October 2026.
v1.10 adds the replication plan of the main comparison: config B3-ovlonly-L (the overlap loss without the containment
term, for attribution) and seed 1 of B3-L and B3-ovl-L (run with --seed 1); see RULES["replication_plan"].
From v1.9:
v1.9 adds the label-relation decoder (LRD; configs B3-lrd-L and, as ablation, B3-lrd-noprior-L; see make_vit_lrd):
one query per class, cross-attention to the pyramid, class-to-class self-attention biased by the directed pixel
overlap of the classes measured on the fit part, query masks added to the UPerNet logits, query presence logits and
deep supervision. Everything else is unchanged from v1.8, so runs of v1.8 load and evaluate as before.

Steps (Colab, CPU runtime is enough for both):
  s0b  two extra data measurements: directed containment shares (decides the optional containment term by the S1
       rule) and polygon widths in pixels at 512 / 1024 / 2048 px; plus a check of the width proxy 2*area/perimeter
       against a skeleton-based width on a sample of polygons
  s2   data index (sizes, orientation, classes, thin-class boxes), fit / hold-out split of the training images,
       loader check with timing, and an example figure
  run  (GPU) trains one model, keeps every epoch checkpoint, selects the best epoch on the hold-out part, then evaluates
       the hold-out part and the validation split with every S1 metric and saves the predicted masks and probabilities
       of every image and an example figure. If Colab stops, the same command continues where it stopped; a finished
       run is never repeated.
  speed (GPU, about 2 min) times the data loader and one training step with each loss (v1.5)
  eval (GPU) re-evaluates a finished run from its stored best checkpoint, without training, and adds the metrics of
       this version (toolkit-protocol score); earlier result files are kept.
  final (GPU, v1.8; an L4 is enough) the final result of a finished run, without training, the same for every model:
       chooses the weights on the hold-out part (best epoch, or the mean weights of the last 3, 5 or 8 epochs),
       predicts with test-time augmentation and evaluates the validation split once at threshold 0.5 (main result).
       Secondary result: one threshold per class chosen on the hold-out part, reported only if a 2-fold check inside
       the hold-out part shows a gain. Saves the chosen weights, all predictions and final_v1_8.json.
Every step also writes the pre-specified rules RULES (S1) to <root>/paperc/s1_rules.json; run and eval also write
their printed output to log.txt in the run folder.

  !python pipeline.py s0b
  !python pipeline.py s2
  !python pipeline.py run --model B1
  !python pipeline.py eval --run B1-s0
  !python pipeline.py run --model B3 --epochs E
  !python pipeline.py run --model B3-ovl --ovl-weight 0.3 --epochs E
  !python pipeline.py run --model Full --stage1 B3-ovl0.3-s0 --skel T --epochs E2
  !python pipeline.py final --run B3-ovl-L0.1-s0
  v1.10, replication (A100, about 4 h each, then the final step of each):
  !python pipeline.py run --model B3-L --epochs 20 --seed 1
  !python pipeline.py run --model B3-ovl-L --ovl-weight 0.1 --epochs 20 --seed 1
  !python pipeline.py run --model B3-ovlonly-L --ovl-weight 0.1 --epochs 20
  v1.9, label-relation decoder (A100; about 4.5 h, then the final step):
  !python pipeline.py run --model B3-lrd-L --ovl-weight 0.1 --epochs 20
  !python pipeline.py final --run B3-lrd-L0.1-s0

Data: the dev-phase zip kept on Drive by S0 (<root>/dacl10k/dacl10k_v2_devphase.zip, or a copy of the same size in
another MyDrive/*/dacl10k/ folder) is copied to the local disk and unpacked once per Colab session (about 5-8 min);
without a copy on Drive it is downloaded from the dacl10k AWS link. Nothing on Drive is deleted or overwritten
except this step's own output files in <root>/paperc/.
"""
import argparse, glob, io, json, os, shutil, sys, time, urllib.request, warnings, zipfile
import numpy as np

VERSION = 'pipeline.py'
CLASSES = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity', 'Spalling',
           'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 'Bearing', 'EJoint', 'Drainage', 'PEquipment',
           'JTape', 'WConccor']
C = len(CLASSES); CI = {c: i for i, c in enumerate(CLASSES)}
URL = 'https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip'
ZIP_NAME, ZIP_SIZE = 'dacl10k_v2_devphase.zip', 5109737241
ZIP_SHA256 = 'dcbcd5fb82699076a2c7f3a72492a9ef798870e0ca1f0c9399360f273ea95260'

# ------------------------------------------------------------------------------------------- S1: pre-specified rules
# Fixed on 10 October 2026, after S0 and the literature review, before S0b and before any training.
RULES = {
    'version': 'S1 v1.10', 'date': '12 October 2026',
    # v1.10, fixed before any of these runs: replication of the main comparison and attribution of the overlap term
    'replication_plan': {'seed_1': 'B3-L-s1 and B3-ovl-L0.1-s1 with the same settings as seed 0 (seed changes the head '
                                   'initialisation, data order and crops); primary endpoint: exact label-set match after the '
                                   'final step; report the difference per seed and the mean over the two seeds',
                         'overlap_only': 'B3-ovlonly-L0.1-s0: overlap loss without the containment term, compared with B3-L-s0 '
                                         '(effect of the overlap term alone) and with B3-ovl-L0.1-s0 (effect of the containment term)',
                         'order': 'seed 1 pair first, then overlap-only', 'fixed_on': '12 October 2026'},
    # v1.9, fixed before any LRD training: plan for the label-relation decoder (configs B3-lrd-L, B3-lrd-noprior-L)
    'lrd_plan': {'primary': 'H4: B3-lrd-L0.1-s0 against B3-ovl-L0.1-s0 (both after the v1.8 final step: weights chosen on '
                            'the hold-out part, TTA, threshold 0.5), validation mIoU at full size; supported if the 95% '
                            'paired-bootstrap interval of the difference lies above zero',
                 'secondary': ['exact label-set match', 'pair IoU ExposedRebars & Rust', 'T mIoU', 'mIoU 512 x 512'],
                 'ablation': 'H5: B3-lrd-L0.1-s0 against B3-lrd-noprior-L0.1-s0 (same settings, prior replaced by the '
                             'identity): exact label-set match; run only if H4 is supported or the means are within 0.005',
                 'settings': 'same data, crops, schedule (20 epochs), seed 0, overlap weight 0.1, containment rule and '
                             'final step as B3-ovl-L; prior = directed pixel overlap of the fit part only',
                 'fixed_on': '12 October 2026'},
    'data': 'dacl10k v2 dev-phase zip (sha256 ' + ZIP_SHA256[:16] + '...), 19 classes, multi-label polygons',
    'test_split': 'official validation split (975 images), evaluated once per final model',
    'holdout_share': 0.10, 'holdout_seed': 0,
    'holdout_method': 'iterative multi-label stratification on class presence (training images only)',
    'thin_rel_width': 0.01,                  # polygon is thin if 2*area/perimeter < 1% of the image long side
    'thin_class_min_share': 0.40,            # class enters T if >= 40% of its training polygons are thin (S0)
    'T_classes': ['Crack', 'Cavity', 'ExposedRebars', 'Rust'],          # result of the rule on S0 training data
    'rare_classes': ['WConccor', 'Rockpocket', 'ACrack', 'EJoint', 'Restformwork', 'Bearing'],  # S0 lower third
    'containment_child': 'ExposedRebars', 'containment_parents': ['Rust', 'Spalling'],
    'containment_min_directed_share': 0.80,  # term used only if |ER and (Rust or Spalling)| / |ER| >= 0.80 (train)
    'global_long_side': 1024, 'train_crop': 768, 'scale_jitter': [0.75, 1.25],
    'refine_long_side': 2048, 'refine_tile': 512, 'refine_no_upscale': True, 'tile_T_share': 0.5,
    'threshold': 0.5,
    'iou': 'per class over the whole evaluation set: sum of intersections / sum of unions; mIoU = mean of 19 classes',
    'eval_resolution': 'native image size (main); 512 x 512 resize (comparison with the dacl10k paper)',
    'topology_eval_long_side': 2048, 'centerline_tolerance_px': 4, 'instance_hit_share': 0.5,
    'seeds_final': [0, 1, 2], 'seeded_models': ['Full', 'B3', 'B3-HR'], 'bootstrap_resamples': 2000,
    'loss_weights': {'presence': 0.1, 'skeleton_recall': 1.0, 'overlap_grid': [0.1, 0.3, 1.0], 'containment': 1.0},
    'augmentation': 'same for all models: horizontal flip, scale jitter, mild colour jitter',
    'model_selection': 'checkpoint with the best hold-out mIoU; the validation split is not used for any choice',
    'primary_comparison': 'full model vs B3-HR (same backbone trained and tested at long side 2048) on T-class mIoU',
    'tier1_T4': ['B1', 'B1-HR', 'B3', 'B3-HR', 'B5', 'Full', 'A3 (skeleton loss)', 'A4 (overlap loss)'],
    'tier2_Pro': ['B2 SegFormer', 'B4 Mask2Former', 'B6 DINOv2-B', 'ConvNeXt-B versions', 'A2', 'A5', 'A6', 'A7'],
    # v1.1 (10 October 2026, before any training): training details and three clarifications
    'epochs_B1': 30,                         # the dacl10k paper's 30 epochs; other models after the B1 timing
    'optimizer': {'name': 'AdamW', 'lr_encoder': 1e-4, 'lr_head': 1e-3, 'weight_decay': 1e-4, 'warmup_epochs': 1,
                  'poly_power': 0.9, 'grad_clip': 1.0, 'precision': 'mixed (float16)'},
    'selection_metric': 'hold-out mIoU at long side 1024 after every epoch',
    'b5_priority': ['Cavity', 'ExposedRebars', 'Crack', 'Rust', 'Drainage', 'Restformwork', 'JTape', 'Spalling',
                    'Rockpocket', 'Efflorescence', 'Bearing', 'WConccor', 'Graffiti', 'Wetspot', 'PEquipment', 'EJoint',
                    'Hollowareas', 'ACrack', 'Weathering'],      # single label for B5: smallest S0 median area wins
    'refine_select_prob': 0.1,               # inference: refine a tile if any T-class probability exceeds 0.1
    'full_model': 'stage 1: global path with overlap (+ containment) loss; stage 2: refinement head on the frozen global path',
    'saved_predictions': 'every hold-out and validation image: masks at long side 1024 (bit-packed), probabilities at long side 512 (uint8)',
    'epochs': {'B1': 30, 'B1-HR': 30, 'B3': 20, 'B3-HR': 20, 'B5': 20, 'B3-ovl': 20, 'overlap_screening': 10,
               'Full_stage2': 10},       # v1.2: fixed from the B1 timing (S1 section 7)
    'screening_patience': 3,                 # v1.3: screening runs stop after 3 epochs without hold-out improvement
    'toolkit_protocol': 'labels as built by the dacl10k toolkit: full-size masks resized to 512 x 512 with '
                        'skimage.transform.resize (anti-aliasing, bilinear), every positive pixel counts; '
                        'prediction at 512 x 512; validation split only; for comparison with published numbers',
    'changes': ['v1.1: A3 clDice variant on Crack only (ACrack is not in T, so it is not refined)',
                'v1.1: B5 single-label target rule; optimiser and schedule details; selection at long side 1024',
                'v1.1: containment term on by the S0b result (98.2% of ExposedRebars pixels inside Rust or Spalling)',
                'v1.2: containment term only in the proposed model (stage 1, B3-ovl); B1 ran with v1.1, where the '
                'term was also active (loss value at most 0.0008), and is reported as run',
                'v1.2: epochs fixed from the B1 timing; toolkit-protocol score added for comparison',
                'v1.3: early stopping for the overlap-weight screening runs only (3 epochs without hold-out '
                'improvement), the same for all three weights; full-length runs keep their fixed epochs',
                'v1.4: bug fix, no rule change: all losses in 32-bit outside mixed precision (in v1.3 the overlap '
                'term overflowed 16-bit and gave NaN); runs with the overlap term made by v1.3 or earlier are invalid, '
                'kept under a new folder name and started again; a run stops at once if its weights become NaN',
                'v1.5: speed only, no rule change: the overlap term is computed with the same formula on the class '
                'pairs that overlap, without 19 x 19 matrix products over all pixels; a run started with v1.4 may be '
                'continued with v1.5 (both versions are recorded in its config.json)',
                'v1.6: bug fix: the Dice loss now follows the multilabel Dice of segmentation_models_pytorch, the library '
                'of the dacl10k paper: a class without labelled pixels in a batch adds 0. Up to v1.5 such a class added '
                'a term that pushed its predictions to 0 everywhere, so classes absent from most batches were never '
                'predicted (B1: IoU 0 for 10 of 19 classes). All runs are repeated with v1.6, including the overlap-weight '
                'screening; runs made with earlier versions are kept under a new folder name',
                'v1.7: large-backbone tier (planned as tier 2): DINOv3 ViT-L/16 (timm vit_large_patch16_dinov3.lvd1689m) '
                'with a UPerNet decoder on a 4-level feature pyramid built from blocks 6, 12, 18 and 24 (as in the best '
                'public single model on dacl10k); runs B3-L (baseline) and B3-ovl-L (our stage 1, overlap weight from the '
                'ConvNeXt-T screening); bfloat16 mixed precision, backbone LR 2e-5, head LR 4e-4, weight decay 0.05, '
                'drop path 0.1; optional test-time augmentation (scales 0.75, 1, 1.25 and horizontal flip) in the eval '
                'step, reported next to the plain score',
                'v1.8 (written while B3-L and B3-ovl-L were training, before their final results): a final step for '
                'finished runs, the same for every model and chosen on the hold-out part only. (1) Weights: the best '
                'epoch, or the mean of the weights of the last 3, 5 or 8 epochs, whichever has the highest hold-out mIoU '
                'at long side 1024 (threshold 0.5, no TTA). (2) Test-time augmentation always on. (3) The main result keeps '
                'the threshold 0.5 for every class (section 8). As a secondary result only, one threshold per class from '
                '0.10 to 0.90 in steps of 0.05 is chosen on the hold-out part and reported if a 2-fold check inside the '
                'hold-out part (choose on one half, score on the other) gives a mean gain over 0.5. The validation split '
                'is evaluated once at the end; the plain best-epoch result of the run stays reported next to it'],
    'tta_scales': [0.75, 1.0, 1.25],
    'final_weight_candidates': {'best': 0, 'last3': 3, 'last5': 5, 'last8': 8},       # v1.8
    'final_threshold_grid': [round(0.10 + 0.05 * i, 2) for i in range(17)],          # v1.8: 0.10 ... 0.90
    'final_threshold_bins': 100, 'final_threshold_folds': 2,
    'final_threshold_rule': 'main result at 0.5 for every class; per-class thresholds (chosen on the hold-out part) '
                            'reported as a secondary result only if the mean 2-fold gain inside the hold-out part is > 0',
}


LOG_PATH = None


def log(*a):
    print(*a, flush=True)
    if LOG_PATH:
        try:
            with open(LOG_PATH, 'a') as f: f.write(' '.join(str(x) for x in a) + '\n')
        except OSError: pass


def save_json(obj, path):
    tmp = path + '.part'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------- data setup
def find_split_dir(root, kind, split):
    hits = [p for p in glob.glob(os.path.join(root, '**', kind, split), recursive=True) if os.path.isdir(p)]
    return sorted(hits, key=len)[0] if hits else None


def setup_data(a):
    """unpacked dataset folder on the local disk (contains annotations/ and images/)."""
    if a.data: return a.data
    os.makedirs(a.work, exist_ok=True)
    if find_split_dir(a.work, 'annotations', 'train') and find_split_dir(a.work, 'images', 'validation'):
        log('dataset already unpacked in', a.work); return a.work
    zlocal = os.path.join(a.work, ZIP_NAME)
    if not (os.path.exists(zlocal) and os.path.getsize(zlocal) == ZIP_SIZE):
        cands = [a.zip] if a.zip else [os.path.join(a.root, 'dacl10k', ZIP_NAME)] + sorted(
            glob.glob(os.path.join(a.mydrive, '*', 'dacl10k', ZIP_NAME)))
        src = next((p for p in cands if p and os.path.isfile(p) and (a.zip or os.path.getsize(p) == ZIP_SIZE)), None)
        if src:
            log(f'copying {src} to the local disk ...'); t = time.time()
            shutil.copy(src, zlocal + '.part'); os.replace(zlocal + '.part', zlocal); log(f'  {time.time() - t:.0f} s')
        else:
            log(f'no copy on Drive; downloading {URL} ...')
            urllib.request.urlretrieve(URL, zlocal + '.part'); os.replace(zlocal + '.part', zlocal)
    log('unpacking ...'); t = time.time()
    dest = os.path.join(a.work, 'v2')
    zipfile.ZipFile(zlocal).extractall(dest + '.part'); os.replace(dest + '.part', dest)
    log(f'  {time.time() - t:.0f} s'); return dest


def annotations(root, split):
    adir = find_split_dir(root, 'annotations', split)
    if adir is None: sys.exit(f'no annotations/{split} below {root}')
    return sorted(glob.glob(os.path.join(adir, '*.json')))


def image_path(root, split, ann, json_path):
    idir = find_split_dir(root, 'images', split)
    name = ann.get('imageName') or os.path.splitext(os.path.basename(json_path))[0] + '.jpg'
    return os.path.join(idir, os.path.basename(name))


def polygons(ann):
    """[(class index, (n, 2) float array in original pixels)] for polygons and rectangles."""
    out = []
    for sh in ann['shapes']:
        if sh['label'] not in CI: continue
        p = np.asarray(sh['points'], float)
        if sh.get('shape_type', 'polygon') == 'rectangle' and len(p) == 2:
            (x0, y0), (x1, y1) = p; p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if len(p) >= 3: out.append((CI[sh['label']], p))
    return out


def area_perimeter(p):
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area, np.sqrt((np.diff(np.vstack([p, p[:1]]), axis=0) ** 2).sum(1)).sum()


def rasterize_window(polys, scale, x0, y0, w, h, classes=None):
    """(C, h, w) bool masks of the window [x0, x0 + w) x [y0, y0 + h) of the image scaled by `scale` (a number or
    (x scale, y scale))."""
    from PIL import Image, ImageDraw
    masks = np.zeros((C, h, w), bool); layers = {}
    for c, p in polys:
        if classes is not None and c not in classes: continue
        q = p * scale - (x0, y0)
        if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() >= w or q[:, 1].min() >= h: continue
        img = layers.get(c)
        if img is None: img = layers[c] = Image.new('1', (w, h), 0)
        ImageDraw.Draw(img).polygon([tuple(v) for v in q], fill=1, outline=1)
    for c, img in layers.items(): masks[c] = np.asarray(img, bool)
    return masks


def load_image(path, W, H, scale, window=None):
    """RGB uint8 image of the annotated size (W, H) scaled by `scale`. EXIF orientation is applied only when the stored
    size is the transposed annotation size; JPEG draft mode makes downscaled decoding fast. With window = (x0, y0, w, h)
    in scaled pixels only that part is resized and returned, zero-padded outside the image, with its valid mask."""
    from PIL import Image, ImageOps
    sx, sy = (scale, scale) if np.isscalar(scale) else scale             # one scale, or (x scale, y scale)
    ws, hs = max(1, round(W * sx)), max(1, round(H * sy))
    with Image.open(path) as im:
        if im.size != (W, H) and im.size == (H, W): im = ImageOps.exif_transpose(im)
        if im.size == (W, H) and im.format == 'JPEG': im.draft('RGB', (ws, hs))
        im = im.convert('RGB')
        if window is None: return np.asarray(im.resize((ws, hs), Image.BILINEAR))
        x0, y0, w, h = window
        out = np.zeros((h, w, 3), np.uint8); valid = np.zeros((h, w), bool)
        xs, ys, xe, ye = max(0, x0), max(0, y0), min(ws, x0 + w), min(hs, y0 + h)
        if xe > xs and ye > ys:
            fx, fy = im.size[0] / ws, im.size[1] / hs                    # source pixels per scaled pixel
            part = im.resize((xe - xs, ye - ys), Image.BILINEAR, box=(xs * fx, ys * fy, xe * fx, ye * fy))
            out[ys - y0:ye - y0, xs - x0:xe - x0] = np.asarray(part); valid[ys - y0:ye - y0, xs - x0:xe - x0] = True
        return out, valid


def skeleton(mask):
    """centre line as the ridge of the Euclidean distance transform (local maxima in a 3 x 3 window). scipy only,
    fast (one distance transform), and the same everywhere; the scikit-image skeleton crashed on some masks."""
    from scipy.ndimage import distance_transform_edt, maximum_filter
    m = mask.astype(bool)
    if not m.any(): return m
    d = distance_transform_edt(m)
    return m & (d >= maximum_filter(d, size=3))


def skeleton_tubes(mask, radius=1):
    """skeleton of a bool mask widened to a tube (Skeleton Recall targets)."""
    from scipy.ndimage import binary_dilation
    if not mask.any(): return mask.copy()
    return binary_dilation(skeleton(mask), np.ones((2 * radius + 1,) * 2, bool))


def sample_global(meta, rng, train=True):
    """global-path sample: image at long side 1024 (x scale jitter when training) and a random train_crop window;
    whole image when not training. Returns image (h, w, 3), masks (19, h, w), valid (h, w)."""
    W, H = meta['W'], meta['H']; L = RULES['global_long_side']
    s = L / max(W, H) * (rng.uniform(*RULES['scale_jitter']) if train else 1.0)
    ws, hs = max(1, round(W * s)), max(1, round(H * s))
    if train:
        cs = RULES['train_crop']
        x0 = int(rng.integers(0, ws - cs + 1)) if ws > cs else 0; y0 = int(rng.integers(0, hs - cs + 1)) if hs > cs else 0
        win = (x0, y0, cs, cs)
    else: win = (0, 0, ws, hs)
    img, valid = load_image(meta['image'], W, H, s, win)
    masks = rasterize_window(polygons(json.load(open(meta['json']))), s, *win)
    return img, masks & valid, valid


def sample_tile(meta, rng):
    """refinement-path sample: tile at long side min(2048, native); centred on a thin-class polygon with
    probability tile_T_share, else uniform."""
    W, H = meta['W'], meta['H']; L = RULES['refine_long_side']
    s = min(L, max(W, H)) / max(W, H) if RULES['refine_no_upscale'] else L / max(W, H)
    T = RULES['refine_tile']; ws, hs = max(1, round(W * s)), max(1, round(H * s))
    if meta['tboxes'] and rng.random() < RULES['tile_T_share']:
        c, bx0, by0, bx1, by1 = meta['tboxes'][int(rng.integers(len(meta['tboxes'])))]
        cx, cy = rng.uniform(bx0, bx1) * s, rng.uniform(by0, by1) * s
    else:
        cx, cy = rng.uniform(0, ws), rng.uniform(0, hs)
    x0 = int(np.clip(cx - T / 2, min(0, ws - T), max(0, ws - T))); y0 = int(np.clip(cy - T / 2, min(0, hs - T), max(0, hs - T)))
    img, valid = load_image(meta['image'], W, H, s, (x0, y0, T, T))
    masks = rasterize_window(polygons(json.load(open(meta['json']))), s, x0, y0, T, T)
    return img, masks & valid, valid


# ---------------------------------------------------------------------------------------------------- S0b
def step_s0b(a, root):
    """directed shares, widths in pixels at 512/1024/2048, width-proxy check."""
    L = RULES['global_long_side']; Tset = [CI[c] for c in RULES['T_classes']]
    ER, PAR = CI[RULES['containment_child']], [CI[c] for c in RULES['containment_parents']]
    out = {'version': VERSION, 'rules': RULES['version'], 'raster_long_side': L, 'splits': {}}
    rng = np.random.default_rng(0)
    for split in ('train', 'validation'):
        files = annotations(root, split)
        if a.max_images: files = files[:a.max_images]
        inter = np.zeros((C, C)); area = np.zeros(C); er_in_par = 0.0; er_img_shares = []
        widths = [[] for _ in range(C)]                                    # 2A/P relative to the long side
        sample = {c: [] for c in Tset}
        for n, jp in enumerate(files):
            ann = json.load(open(jp)); W, H = ann['imageWidth'], ann['imageHeight']; ps = polygons(ann)
            for c, p in ps:
                ar, pe = area_perimeter(p)
                if pe > 0: widths[c].append(2 * ar / pe / max(W, H))
                if split == 'train' and c in sample and ar > 0: sample[c].append((jp, W, H, p, 2 * ar / pe))
            if not ps: continue
            s = L / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
            m = rasterize_window(ps, s, 0, 0, w, h)
            flat = m.reshape(C, -1); ar = flat.sum(1); area += ar
            idx = np.flatnonzero(ar)
            for i in idx:
                inter[i, idx] += (flat[i] & flat[idx]).sum(1)
            if ar[ER]:
                par = flat[PAR].any(0); k = (flat[ER] & par).sum(); er_in_par += k; er_img_shares.append(k / ar[ER])
            if (n + 1) % 1000 == 0: log(f'  {split}: {n + 1} / {len(files)}')
        with np.errstate(invalid='ignore', divide='ignore'): directed = inter / area[:, None]   # |a and b| / |a|
        er_share = er_in_par / area[ER] if area[ER] else float('nan')
        px = {}
        for Lt in (512, 1024, 2048):
            px[Lt] = {CLASSES[c]: {f'<{k}px': round(100 * float((np.asarray(widths[c]) * Lt < k).mean()), 1)
                                   for k in (1, 2, 3)} for c in range(C) if widths[c]}
        out['splits'][split] = {'directed_share': directed, 'er_in_rust_or_spalling': er_share,
                                'er_per_image_median': float(np.median(er_img_shares)) if er_img_shares else None,
                                'width_px_shares': px}
        log(f'\n=== {split}: {len(files)} images; directed pixel shares |a and b| / |a| at long side {L}')
        log(f'ExposedRebars inside Rust or Spalling: {100 * er_share:.1f}% of its pixels '
            f'(median per image {100 * np.median(er_img_shares):.1f}%)')
        for b in PAR + [CI["Cavity"]]:
            log(f'  ExposedRebars -> {CLASSES[b]}: {100 * directed[ER, b]:.1f}%   |   {CLASSES[b]} -> ExposedRebars: '
                f'{100 * directed[b, ER]:.1f}%')
        top = sorted(((directed[i, j], i, j) for i in range(C) for j in range(C) if i != j and area[i] > 0),
                     reverse=True)[:12]
        log('largest directed shares:', ', '.join(f'{CLASSES[i]}->{CLASSES[j]} {100 * v:.1f}%' for v, i, j in top))
        log('\nshare of polygons narrower than 1 / 2 / 3 px (width = 2*area/perimeter, scaled to the long side):')
        log(f'{"class":15s}' + ''.join(f'{f"{Lt} px":>18s}' for Lt in (512, 1024, 2048)))
        for c in range(C):
            if not widths[c]: continue
            log(f'{CLASSES[c]:15s}' + ''.join(
                f'{px[Lt][CLASSES[c]]["<1px"]:6.1f}{px[Lt][CLASSES[c]]["<2px"]:6.1f}{px[Lt][CLASSES[c]]["<3px"]:6.1f}'
                for Lt in (512, 1024, 2048)))
        if split == 'train':
            use = er_share >= RULES['containment_min_directed_share']
            out['containment_term_used'] = bool(use)
            log(f'\nS1 rule: containment term used if the share >= {RULES["containment_min_directed_share"]:.2f} '
                f'-> {"USED" if use else "NOT USED"} ({er_share:.3f})')
            out['width_proxy_check'] = width_check(sample, rng, a.width_sample)
    save_json(out, os.path.join(a.out, 's0b_stats.json')); log('\nwritten', os.path.join(a.out, 's0b_stats.json'))


def width_check(sample, rng, n):
    """2*area/perimeter vs. centre-line width (2 x mean distance to the boundary on the centre line - 0.5 px), native resolution."""
    from scipy.ndimage import distance_transform_edt
    from scipy.stats import spearmanr
    res = {}
    log(f'\nwidth proxy check (native resolution, up to {n} random training polygons per thin class):')
    for c, items in sample.items():
        if not items: continue
        pick = rng.choice(len(items), size=min(n, len(items)), replace=False)
        prox, skel = [], []
        for k in pick:
            _, W, H, p, wp = items[k]
            x0, y0 = np.floor(p.min(0)).astype(int) - 2; x1, y1 = np.ceil(p.max(0)).astype(int) + 3
            if (x1 - x0) * (y1 - y0) > 4e6: continue                          # skip very large polygons
            m = rasterize_window([(c, p)], 1.0, x0, y0, x1 - x0, y1 - y0)[c]
            if m.sum() < 4: continue
            sk = skeleton(m); d = distance_transform_edt(m)
            if not sk.any(): continue
            prox.append(wp); skel.append(2 * d[sk].mean() - 0.5)            # -0.5: odd/even pixel widths
        prox, skel = np.asarray(prox), np.asarray(skel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); r = spearmanr(prox, skel)
        rho = float(r.correlation if hasattr(r, 'correlation') else r[0]); rho = None if np.isnan(rho) else rho
        ratio = float(np.median(prox / skel))
        res[CLASSES[c]] = {'n': len(prox), 'spearman': rho, 'median_ratio_proxy_to_skeleton': ratio,
                           'median_proxy_px': float(np.median(prox)), 'median_skeleton_px': float(np.median(skel))}
        log(f'  {CLASSES[c]:15s} n {len(prox):4d}  Spearman {"n/a" if rho is None else f"{rho:.2f}"}  median proxy/centre-line {ratio:.2f}  '
            f'median width proxy {np.median(prox):.1f} px, centre-line {np.median(skel):.1f} px')
    return res


# ---------------------------------------------------------------------------------------------------- S2
def iterative_split(Y, share, seed):
    """iterative stratification (Sechidis et al. 2011) into two parts [1 - share, share]; returns 0/1 per image."""
    rng = np.random.default_rng(seed); N = len(Y)
    r = np.array([1 - share, share]); desired = np.outer(r, Y.sum(0)).astype(float); total = r * N
    assign = -np.ones(N, int); remaining = np.ones(N, bool)
    while remaining.any():
        cnt = Y[remaining].sum(0)
        if cnt.sum() == 0:
            for i in np.flatnonzero(remaining):
                k = int(np.argmax(total)); assign[i] = k; total[k] -= 1; remaining[i] = False
            break
        c = int(np.argmin(np.where(cnt > 0, cnt, np.inf)))
        idx = np.flatnonzero(remaining & Y[:, c]); rng.shuffle(idx)
        for i in idx:
            d = desired[:, c]; best = np.flatnonzero(d == d.max())
            if len(best) > 1: best = best[total[best] == total[best].max()]
            k = int(rng.choice(best)); assign[i] = k; remaining[i] = False; desired[k] -= Y[i]; total[k] -= 1
    return assign


def build_index(root, max_images=0):
    from PIL import Image
    Tset = {CI[c] for c in RULES['T_classes']}; index = []
    issues = {'missing_image': 0, 'transposed': 0, 'other_size': 0, 'exif_orientation_not_1': 0}
    exts = {}
    for split in ('train', 'validation'):
        files = annotations(root, split)
        if max_images: files = files[:max_images]
        for jp in files:
            ann = json.load(open(jp)); W, H = ann['imageWidth'], ann['imageHeight']; ip = image_path(root, split, ann, jp)
            exts[os.path.splitext(ip)[1].lower()] = exts.get(os.path.splitext(ip)[1].lower(), 0) + 1
            orient = 'same'
            if not os.path.exists(ip): issues['missing_image'] += 1; orient = 'missing'
            else:
                with Image.open(ip) as im:
                    if im.getexif().get(0x0112, 1) != 1: issues['exif_orientation_not_1'] += 1
                    if im.size == (H, W) and W != H: orient = 'transposed'; issues['transposed'] += 1
                    elif im.size != (W, H): orient = 'other'; issues['other_size'] += 1
            ps = polygons(ann); pres = np.zeros(C, bool); npoly = np.zeros(C, int); thin = np.zeros(C, int); tboxes = []
            for c, p in ps:
                pres[c] = True; npoly[c] += 1; ar, pe = area_perimeter(p)
                if pe > 0 and 2 * ar / pe < RULES['thin_rel_width'] * max(W, H): thin[c] += 1
                if c in Tset:
                    x0, y0 = p.min(0); x1, y1 = p.max(0); tboxes.append([c, round(float(x0), 1), round(float(y0), 1),
                                                                         round(float(x1), 1), round(float(y1), 1)])
            index.append({'split': split, 'name': os.path.basename(ip), 'json': jp, 'image': ip, 'W': W, 'H': H,
                          'orient': orient, 'presence': pres.astype(int).tolist(), 'npoly': npoly.tolist(),
                          'nthin': thin.tolist(), 'tboxes': tboxes})
    return index, issues, exts


def example_figure(index, path, rng):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt, matplotlib.colors, matplotlib.patches
    Tcols = {'Crack': '#d62728', 'Cavity': '#ff7f0e', 'ExposedRebars': '#9467bd', 'Rust': '#1f77b4'}
    cand = [m for m in index if m['split'] == 'train' and m['orient'] != 'missing'
            and sum(m['presence'][CI[c]] for c in Tcols) >= 2]
    picks = [cand[i] for i in rng.choice(len(cand), size=min(3, len(cand)), replace=False)]
    fig, axs = plt.subplots(len(picks), 3, figsize=(10.5, 3.0 * len(picks)), squeeze=False)
    for r, m in enumerate(picks):
        img, masks, valid = sample_global(m, rng, train=False)
        tile, tmasks, tvalid = sample_tile(m, np.random.default_rng(r))
        for ax, im, ms, title in ((axs[r, 0], img, masks, 'global path, long side 1024'),
                                  (axs[r, 1], tile, tmasks, 'refinement tile, long side 2048')):
            ax.imshow(im)
            over = np.zeros(im.shape[:2] + (4,))
            for c, col in Tcols.items():
                rgb = matplotlib.colors.to_rgb(col); over[ms[CI[c]]] = (*rgb, 0.45)
            ax.imshow(over); ax.set_title(title, fontsize=8, loc='left'); ax.axis('off')
        sk = np.zeros(tile.shape[:2] + (4,))
        for c, col in Tcols.items():
            sk[skeleton_tubes(tmasks[CI[c]])] = (*matplotlib.colors.to_rgb(col), 1.0)
        axs[r, 2].imshow(tile, alpha=0.35); axs[r, 2].imshow(sk); axs[r, 2].axis('off')
        axs[r, 2].set_title('skeleton targets (thin classes)', fontsize=8, loc='left')
    handles = [matplotlib.patches.Patch(color=v, label=k) for k, v in Tcols.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


def step_s2(a, root):
    rng = np.random.default_rng(0); t = time.time()
    index, issues, exts = build_index(root, a.max_images)
    log(f'index: {len(index)} images in {time.time() - t:.0f} s; file types {exts}; issues {issues}')
    tr = [m for m in index if m['split'] == 'train']
    Y = np.array([m['presence'] for m in tr], bool)
    part = iterative_split(Y, RULES['holdout_share'], RULES['holdout_seed'])
    split = {'rules': RULES['version'], 'fit': [tr[i]['name'] for i in np.flatnonzero(part == 0)],
             'holdout': [tr[i]['name'] for i in np.flatnonzero(part == 1)]}
    log(f'\nfit / hold-out split of the training images: {len(split["fit"])} / {len(split["holdout"])}')
    log(f'{"class":15s}{"train":>7s}{"hold-out":>10s}{"share %":>9s}')
    rows = []
    for c in range(C):
        n, k = int(Y[:, c].sum()), int(Y[part == 1, c].sum())
        rows.append({'class': CLASSES[c], 'train_images': n, 'holdout_images': k})
        log(f'{CLASSES[c]:15s}{n:7d}{k:10d}{100 * k / max(n, 1):9.1f}')
    # loader check and timing (one process; Colab gives 2 CPU workers for training)
    times = {'global': [], 'tile': [], 'skeleton': []}; shapes = {}
    for k in range(a.bench):
        m = tr[int(rng.integers(len(tr)))]
        t0 = time.time(); img, masks, valid = sample_global(m, rng); times['global'].append(time.time() - t0)
        shapes['global'] = [list(img.shape), list(masks.shape)]
        t0 = time.time(); tile, tm, tv = sample_tile(m, rng); times['tile'].append(time.time() - t0)
        shapes['tile'] = [list(tile.shape), list(tm.shape)]
        t0 = time.time()
        for c in RULES['T_classes']: skeleton_tubes(tm[CI[c]])
        times['skeleton'].append(time.time() - t0)
        assert img.shape[:2] == masks.shape[1:] == valid.shape and tile.shape[:2] == tm.shape[1:]
    bench = {k: round(1000 * float(np.mean(v)), 1) for k, v in times.items()}
    per = (bench['global'] + bench['tile'] + bench['skeleton']) / 1000
    log(f'\nloader check on {a.bench} random training images: shapes {shapes}')
    log(f'ms per sample: global {bench["global"]}, tile {bench["tile"]}, skeletons {bench["skeleton"]} '
        f'-> about {2 / per:.1f} samples/s with 2 workers')
    example_figure(index, os.path.join(a.out, 'fig_s2_examples.png'), rng)
    save_json({'version': VERSION, 'rules': RULES['version'], 'n': len(index), 'issues': issues, 'file_types': exts,
               'index': index}, os.path.join(a.out, 's2_index.json'))
    save_json(split, os.path.join(a.out, 's2_split.json'))
    save_json({'version': VERSION, 'issues': issues, 'holdout_rows': rows, 'loader_ms': bench, 'shapes': shapes},
              os.path.join(a.out, 's2_report.json'))
    log('\nwritten s2_index.json, s2_split.json, s2_report.json, fig_s2_examples.png in', a.out)


# ---------------------------------------------------------------------------------------------------- models and runs
# Run configurations (S1 section 5). 'mode' sets the working resolution: fixed512 = whole image stretched to 512 x 512
# (dacl10k paper recipe); long1024 = long side 1024; long2048 = long side min(2048, native).
CONFIGS = {
    'B1':     dict(arch='fpn-efficientnet-b4', mode='fixed512', crop=512, loss='dice', head='sigmoid', batch=8, accum=1),
    'B1-HR':  dict(arch='fpn-efficientnet-b4', mode='long1024', crop=768, loss='dice', head='sigmoid', batch=4, accum=2),
    'B3':     dict(arch='upernet-convnext_tiny', mode='long1024', crop=768, loss='bce+dice', head='sigmoid', batch=4, accum=2),
    'B3-HR':  dict(arch='upernet-convnext_tiny', mode='long2048', crop=768, loss='bce+dice', head='sigmoid', batch=4, accum=2),
    'B5':     dict(arch='upernet-convnext_tiny', mode='long1024', crop=768, loss='ce+dice', head='softmax', batch=4, accum=2),
    'B3-ovl': dict(arch='upernet-convnext_tiny', mode='long1024', crop=768, loss='bce+dice', head='sigmoid', batch=4, accum=2,
                   ovl=True),
    'B3-L':   dict(arch='uper-vit_large_patch16_dinov3.lvd1689m', mode='long1024', crop=768, loss='bce+dice',
                   head='sigmoid', batch=4, accum=2, amp='bf16', lr_encoder=2e-5, lr_head=4e-4, weight_decay=0.05),
    'B3-ovl-L': dict(arch='uper-vit_large_patch16_dinov3.lvd1689m', mode='long1024', crop=768, loss='bce+dice',
                     head='sigmoid', batch=4, accum=2, ovl=True, amp='bf16', lr_encoder=2e-5, lr_head=4e-4,
                     weight_decay=0.05),
    # v1.10: the overlap term alone (no containment term), for attribution
    'B3-ovlonly-L': dict(arch='uper-vit_large_patch16_dinov3.lvd1689m', mode='long1024', crop=768, loss='bce+dice',
                         head='sigmoid', batch=4, accum=2, ovl=True, no_cont=True, amp='bf16', lr_encoder=2e-5, lr_head=4e-4,
                         weight_decay=0.05),
    # v1.9: label-relation decoder; prior = directed pixel overlap of the fit part (B3-lrd-L) or identity (ablation)
    'B3-lrd-L': dict(arch='lrd-vit_large_patch16_dinov3.lvd1689m', mode='long1024', crop=768, loss='bce+dice',
                     head='sigmoid', batch=4, accum=2, ovl=True, amp='bf16', lr_encoder=2e-5, lr_head=4e-4, lr_lrd=2e-4,
                     weight_decay=0.05, lrd_prior=True, aux_w=0.4),
    'B3-lrd-noprior-L': dict(arch='lrd-vit_large_patch16_dinov3.lvd1689m', mode='long1024', crop=768, loss='bce+dice',
                             head='sigmoid', batch=4, accum=2, ovl=True, amp='bf16', lr_encoder=2e-5, lr_head=4e-4,
                             lr_lrd=2e-4, weight_decay=0.05, lrd_prior=False, aux_w=0.4),
    'Full':   dict(stage=2, arch='refine', mode='tile', crop=512, loss='bce+dice', head='sigmoid', batch=8, accum=1),
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255; STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
TIDX = [CI[c] for c in RULES['T_classes']]


def torch_mods():
    import torch, torch.nn as nn, torch.nn.functional as F
    return torch, nn, F


def ensure_smp():
    import importlib.util
    if importlib.util.find_spec('segmentation_models_pytorch') is None:
        import subprocess
        log('installing segmentation-models-pytorch ...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'])


def make_vit_uper(name, n_cls, pretrained=True, ch=256, drop_path=0.1):
    """v1.7: plain ViT backbone (timm, any input size divisible by 16) + UPerNet decoder. Blocks at 1/4, 1/2, 3/4 and the
    end of the ViT are turned into a pyramid at strides 4, 8, 16 and 32 (DPT-style reassembly), then pyramid pooling,
    a top-down FPN and a fuse layer (GroupNorm) give 19 per-pixel logits; an image-level head (19 presence logits)
    sits on the last block, as in the smp models. Returns (logits, presence) like smp with aux_params."""
    import timm
    torch, nn, F = torch_mods()

    def cgr(i, o, k=3):
        return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.GroupNorm(32, o), nn.ReLU(inplace=True))

    class ViTUper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(name, pretrained=pretrained, dynamic_img_size=True, drop_path_rate=drop_path)
            D, n = self.encoder.embed_dim, len(self.encoder.blocks)
            self.idx = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
            self.re = nn.ModuleList([nn.Sequential(nn.Conv2d(D, ch, 1), nn.ConvTranspose2d(ch, ch, 4, 4)),
                                     nn.Sequential(nn.Conv2d(D, ch, 1), nn.ConvTranspose2d(ch, ch, 2, 2)),
                                     nn.Conv2d(D, ch, 1),
                                     nn.Sequential(nn.Conv2d(D, ch, 1), nn.Conv2d(ch, ch, 3, 2, 1))])
            self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(k), nn.Conv2d(ch, ch, 1, bias=False),
                                                    nn.GroupNorm(32, ch), nn.ReLU(inplace=True)) for k in (1, 2, 3, 6)])
            self.ppm_out = cgr(5 * ch, ch)
            self.lat = nn.ModuleList([cgr(ch, ch, 1) for _ in range(3)])
            self.fpn = nn.ModuleList([cgr(ch, ch) for _ in range(3)])
            self.fuse = cgr(4 * ch, ch)
            self.cls = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(ch, n_cls, 1))
            self.aux = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, C))

        def forward(self, x):
            H, W = x.shape[-2:]
            f = self.encoder.forward_intermediates(x, indices=self.idx, output_fmt='NCHW', intermediates_only=True)
            pres = self.aux(f[-1].mean((2, 3)))
            p = [r(v) for r, v in zip(self.re, f)]
            top = p[3]; up = lambda t, sz: F.interpolate(t, size=sz, mode='bilinear', align_corners=False)
            lat = [l(v) for l, v in zip(self.lat, p[:3])] + [self.ppm_out(torch.cat([top] + [up(m(top), top.shape[-2:]) for m in self.ppm], 1))]
            for i in (2, 1, 0): lat[i] = lat[i] + up(lat[i + 1], lat[i].shape[-2:])
            outs = [self.fpn[i](lat[i]) for i in range(3)] + [lat[3]]
            y = self.fuse(torch.cat([outs[0]] + [up(o, outs[0].shape[-2:]) for o in outs[1:]], 1))
            return up(self.cls(y), (H, W)), pres
    return ViTUper()


def make_vit_lrd(name, n_cls, pretrained=True, ch=256, d=256, layers=3, heads=8, drop_path=0.1):
    """v1.9: label-relation decoder (LRD) on the ViT + UPerNet model of make_vit_uper.
    Pixel path: as make_vit_uper (DPT reassembly to strides 4-32, pyramid pooling, top-down FPN, fuse at stride 4) gives
    base logits and a pixel embedding E (d channels, stride 4).
    Class path: 19 learnable queries, one per class (no matching needed: query k always stands for class k). Each of
    `layers` layers applies (1) cross-attention from the queries to the pyramid level of stride 32, 16 and 8 in turn
    (2-D sine positions and a level embedding), (2) class-relation self-attention among the queries with an additive
    bias alpha_h * P + R, where P (buffer `prior`, 19 x 19) is the directed pixel overlap share |a and b| / |a| measured
    on the fit labels (diagonal 1), alpha_h one learnable weight per head (start 2) and R a learnable 19 x 19 bias
    (start 0), and (3) a feed-forward block; all pre-norm. After every layer, mask logits m_k(u) = <MLP(q_k), E(u)> / sqrt(d).
    Output logits = base logits + m of the last layer (both trained from the start, so the decoder adds to the strong
    UPerNet path); presence logit of class k = <w_k, q_k> + b_k from its own query. In training, the forward also returns
    the mask logits of the earlier layers for deep supervision. Returns (logits, presence[, aux list])."""
    import math, timm
    torch, nn, F = torch_mods()

    def cgr(i, o, k=3):
        return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.GroupNorm(32, o), nn.ReLU(inplace=True))

    def sine_pos(h, w, dim, device):
        """2-D sine position code (half of the channels for y, half for x), shape (h * w, dim)."""
        q = dim // 4; f = 1.0 / (10000 ** (torch.arange(q, device=device).float() / q))
        y = torch.arange(h, device=device).float()[:, None] * f; x = torch.arange(w, device=device).float()[:, None] * f
        py = torch.cat([y.sin(), y.cos()], 1)[:, None, :].expand(h, w, 2 * q)
        px = torch.cat([x.sin(), x.cos()], 1)[None, :, :].expand(h, w, 2 * q)
        return torch.cat([py, px], 2).reshape(h * w, 4 * q)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
            self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
            self.rel = nn.MultiheadAttention(d, heads, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.alpha = nn.Parameter(torch.full((heads,), 2.0)); self.R = nn.Parameter(torch.zeros(n_cls, n_cls))

        def forward(self, q, qpos, mem, mpos, prior):
            h = self.n1(q)
            q = q + self.cross(h + qpos, mem + mpos, mem, need_weights=False)[0]
            h = self.n2(q); B = q.shape[0]
            bias = self.alpha.float()[:, None, None] * prior.float()[None] + self.R.float()[None]     # (heads, K, K)
            bias = bias[None].expand(B, -1, -1, -1).reshape(B * heads, n_cls, n_cls).to(h.dtype)
            q = q + self.rel(h + qpos, h + qpos, h, attn_mask=bias, need_weights=False)[0]
            return q + self.ffn(self.n3(q))

        def relation_weights(self, q, qpos, prior):
            """attention of each class query to the others (mean over heads), for analysis only."""
            h = self.n2(q); B = q.shape[0]
            bias = self.alpha.float()[:, None, None] * prior.float()[None] + self.R.float()[None]
            bias = bias[None].expand(B, -1, -1, -1).reshape(B * heads, n_cls, n_cls).to(h.dtype)
            return self.rel(h + qpos, h + qpos, h, attn_mask=bias, need_weights=True, average_attn_weights=True)[1]

    class ViTLRD(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(name, pretrained=pretrained, dynamic_img_size=True, drop_path_rate=drop_path)
            D, n = self.encoder.embed_dim, len(self.encoder.blocks)
            self.idx = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
            self.re = nn.ModuleList([nn.Sequential(nn.Conv2d(D, ch, 1), nn.ConvTranspose2d(ch, ch, 4, 4)),
                                     nn.Sequential(nn.Conv2d(D, ch, 1), nn.ConvTranspose2d(ch, ch, 2, 2)),
                                     nn.Conv2d(D, ch, 1),
                                     nn.Sequential(nn.Conv2d(D, ch, 1), nn.Conv2d(ch, ch, 3, 2, 1))])
            self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(k), nn.Conv2d(ch, ch, 1, bias=False),
                                                    nn.GroupNorm(32, ch), nn.ReLU(inplace=True)) for k in (1, 2, 3, 6)])
            self.ppm_out = cgr(5 * ch, ch)
            self.lat = nn.ModuleList([cgr(ch, ch, 1) for _ in range(3)])
            self.fpn = nn.ModuleList([cgr(ch, ch) for _ in range(3)])
            self.fuse = cgr(4 * ch, ch)
            self.cls = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(ch, n_cls, 1))
            # label-relation decoder; its parameters (names starting with 'lrd.') get their own learning rate
            self.lrd = nn.ModuleDict({
                'embed': nn.Sequential(cgr(ch, ch), nn.Conv2d(ch, d, 1)),
                'mem': nn.ModuleList([nn.Conv2d(ch, d, 1) for _ in range(3)]),
                'level': nn.Embedding(3, d), 'qfeat': nn.Embedding(n_cls, d), 'qpos': nn.Embedding(n_cls, d),
                'layers': nn.ModuleList([Layer() for _ in range(layers)]),
                'norm': nn.LayerNorm(d),
                'mask_mlp': nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d)),
                'pres': nn.Linear(d, n_cls)})
            self.register_buffer('prior', torch.eye(n_cls))          # replaced by the fit-part overlap before training

        def pixel_path(self, x):
            f = self.encoder.forward_intermediates(x, indices=self.idx, output_fmt='NCHW', intermediates_only=True)
            p = [r(v) for r, v in zip(self.re, f)]
            top = p[3]; up = lambda t, sz: F.interpolate(t, size=sz, mode='bilinear', align_corners=False)
            lat = [l(v) for l, v in zip(self.lat, p[:3])] + [self.ppm_out(torch.cat([top] + [up(m(top), top.shape[-2:]) for m in self.ppm], 1))]
            for i in (2, 1, 0): lat[i] = lat[i] + up(lat[i + 1], lat[i].shape[-2:])
            outs = [self.fpn[i](lat[i]) for i in range(3)] + [lat[3]]
            y = self.fuse(torch.cat([outs[0]] + [up(o, outs[0].shape[-2:]) for o in outs[1:]], 1))
            return y, outs

        def memories(self, outs, x):
            L = self.lrd; mems = []
            for j, lev in enumerate((3, 2, 1)):                            # strides 32, 16, 8
                m = L['mem'][j](outs[lev]); hh, ww = m.shape[-2:]
                mems.append((m.flatten(2).transpose(1, 2), (sine_pos(hh, ww, d, x.device) + L['level'].weight[j])[None].to(m.dtype)))
            return mems

        def forward(self, x):
            H, W = x.shape[-2:]; up = lambda t, sz: F.interpolate(t, size=sz, mode='bilinear', align_corners=False)
            y, outs = self.pixel_path(x)
            base = self.cls(y); L = self.lrd; E = L['embed'](y); B = x.shape[0]
            q = L['qfeat'].weight[None].expand(B, -1, -1); qpos = L['qpos'].weight[None].expand(B, -1, -1)
            mems = self.memories(outs, x); masks = []
            for i, layer in enumerate(L['layers']):
                mem, mpos = mems[i % 3]
                q = layer(q, qpos, mem, mpos, self.prior)
                if self.training or i == len(L['layers']) - 1:
                    k = L['mask_mlp'](L['norm'](q))
                    masks.append(torch.einsum('bkd,bdhw->bkhw', k, E) / math.sqrt(d))
            qn = L['norm'](q)
            pres = (qn * L['pres'].weight[None]).sum(-1) + L['pres'].bias[None]
            logits = up(base + masks[-1], (H, W))
            if self.training: return logits, pres, [up(m, (H, W)) for m in masks[:-1]]
            return logits, pres

        def relations(self, x):
            """class-to-class attention of every LRD layer for one batch (analysis only): list of (B, K, K)."""
            y, outs = self.pixel_path(x); L = self.lrd; B = x.shape[0]
            q = L['qfeat'].weight[None].expand(B, -1, -1); qpos = L['qpos'].weight[None].expand(B, -1, -1)
            mems = self.memories(outs, x); res = []
            for i, layer in enumerate(L['layers']):
                mem, mpos = mems[i % 3]
                h = layer.n1(q); q1 = q + layer.cross(h + qpos, mem + mpos, mem, need_weights=False)[0]
                res.append(layer.relation_weights(q1, qpos, self.prior))
                q = layer(q, qpos, mem, mpos, self.prior)
            return res
    return ViTLRD()


def amp_dtype(cfg):
    """v1.7: bfloat16 for the large ViT runs (no loss scaling needed), float16 for all earlier configurations."""
    torch, _, _ = torch_mods()
    return torch.bfloat16 if (cfg or {}).get('amp') == 'bf16' else torch.float16


def build_model(cfg, pretrained=True, small=False):
    """smp model with an image-level presence head (aux) of 19 outputs; softmax head has 19 classes + background."""
    import segmentation_models_pytorch as smp
    w = 'imagenet' if pretrained else None; n = C + 1 if cfg['head'] == 'softmax' else C
    aux = dict(classes=C, dropout=0.2); arch = cfg['arch']
    if small: arch = {'fpn-efficientnet-b4': 'fpn-resnet18', 'upernet-convnext_tiny': 'upernet-convnext_atto',
                      'uper-vit_large_patch16_dinov3.lvd1689m': 'uper-vit_small_patch16_dinov3.lvd1689m',
                      'lrd-vit_large_patch16_dinov3.lvd1689m': 'lrd-vit_small_patch16_dinov3.lvd1689m'}[arch]
    kind, enc = arch.split('-', 1)
    if kind == 'uper': return make_vit_uper(enc, n, pretrained=pretrained and not small)
    if kind == 'lrd': return make_vit_lrd(enc, n, pretrained=pretrained and not small)
    if kind == 'fpn': return smp.FPN(enc, encoder_weights=w, classes=n, aux_params=aux)
    if kind == 'upernet': return smp.UPerNet('tu-' + enc, encoder_weights=w, classes=n, aux_params=aux)
    raise ValueError(arch)


def make_refine_head(n_out=len(TIDX), n_ctx=C, w=48, k=11):
    """light high-resolution head for the thin/small classes. Input: tile RGB and the global logits of all classes
    cropped to the tile; strip convolutions (1 x k and k x 1) follow long thin shapes; output: blended logits
    alpha * refined + (1 - alpha) * global for the T classes, at tile resolution."""
    torch, nn, F = torch_mods()

    def cbr(i, o, s=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Strip(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.h = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False)
            self.v = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c, bias=False)
            self.d = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
            self.n1, self.pw, self.n2 = nn.BatchNorm2d(c), nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c)

        def forward(self, x): return F.relu(x + self.n2(self.pw(F.relu(self.n1(self.h(x) + self.v(x) + self.d(x))))))

    class Refine(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(cbr(3, w, 2), cbr(w, w))                                    # 1/2
            self.d1 = nn.Sequential(cbr(w, 2 * w, 2), Strip(2 * w))                              # 1/4
            self.d2 = nn.Sequential(cbr(2 * w, 4 * w, 2), Strip(4 * w), Strip(4 * w))            # 1/8
            self.u1 = nn.Sequential(cbr(6 * w, 2 * w), Strip(2 * w))                             # 1/4
            self.u0 = nn.Sequential(cbr(3 * w + n_ctx, w), Strip(w), Strip(w))                  # 1/2
            self.out, self.alpha = nn.Conv2d(w, n_out, 1), nn.Conv2d(w, n_out, 1)
            nn.init.constant_(self.alpha.bias, -1.0)                       # start close to the global prediction

        def forward(self, x, ctx):                                         # x (B, 3, T, T); ctx (B, 19, T/2, T/2)
            s0 = self.stem(x); s1 = self.d1(s0); s2 = self.d2(s1)
            u1 = self.u1(torch.cat([F.interpolate(s2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], 1))
            u0 = self.u0(torch.cat([F.interpolate(u1, size=s0.shape[-2:], mode='bilinear', align_corners=False), s0, ctx], 1))
            a = torch.sigmoid(self.alpha(u0)); fused = a * self.out(u0) + (1 - a) * ctx[:, TIDX]
            return F.interpolate(fused, size=x.shape[-2:], mode='bilinear', align_corners=False)
    return Refine()


def run_scale(mode, W, H):
    """(x scale, y scale) of the working resolution."""
    if mode == 'fixed512': return 512 / W, 512 / H
    if mode == 'long1024': s = RULES['global_long_side'] / max(W, H); return s, s
    if mode in ('long2048', 'tile'): s = min(RULES['refine_long_side'], max(W, H)) / max(W, H); return s, s
    raise ValueError(mode)


# ---------------------------------------------------------------------------------------------------- datasets
def single_label(masks):
    """one label per pixel for the softmax model B5: the class with the smallest S0 median area share wins; 0 = none."""
    lab = np.zeros(masks.shape[1:], np.int64)
    for c in reversed(RULES['b5_priority']):                               # paint large classes first, small last
        lab[masks[CI[c]]] = CI[c] + 1
    return lab


def colour_jitter(img, rng):
    x = img.astype(np.float32)
    x = (x - x.mean()) * rng.uniform(0.8, 1.2) + x.mean() * rng.uniform(0.8, 1.2)
    return np.clip(x, 0, 255).astype(np.uint8)


def to_tensor_img(img):
    torch, _, _ = torch_mods()
    return torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())


def make_train_set(metas, cfg, seed):
    torch, _, _ = torch_mods()

    class TrainSet(torch.utils.data.Dataset):
        def __init__(self): self.metas, self.epoch = metas, 0

        def __len__(self): return len(self.metas)

        def __getitem__(self, i):
            rng = np.random.default_rng([seed, self.epoch, i])
            m = self.metas[i]; W, H = m['W'], m['H']; sx, sy = run_scale(cfg['mode'], W, H)
            j = rng.uniform(*RULES['scale_jitter']); sx, sy = sx * j, sy * j
            ws, hs = max(1, round(W * sx)), max(1, round(H * sy)); cs = cfg['crop']
            x0 = int(rng.integers(0, ws - cs + 1)) if ws > cs else 0; y0 = int(rng.integers(0, hs - cs + 1)) if hs > cs else 0
            img, valid = load_image(m['image'], W, H, (sx, sy), (x0, y0, cs, cs))
            masks = rasterize_window(polygons(json.load(open(m['json']))), (sx, sy), x0, y0, cs, cs) & valid
            if rng.random() < 0.5: img, masks, valid = img[:, ::-1], masks[:, :, ::-1], valid[:, ::-1]
            img = colour_jitter(img, rng)
            out = {'img': to_tensor_img(img), 'valid': torch.from_numpy(valid.copy()),
                   'presence': torch.from_numpy(masks.reshape(C, -1).any(1).astype(np.float32))}
            if cfg['head'] == 'softmax': out['label'] = torch.from_numpy(single_label(masks))
            out['masks'] = torch.from_numpy(masks.copy())
            return out
    return TrainSet()


def make_tile_set(metas, seed, skel):
    """stage-2 samples: a tile at long side min(2048, native) and the whole image at long side 1024 (padded to
    1024 x 1024) for the frozen global path, with the tile box in global-image pixels."""
    torch, _, _ = torch_mods()
    L, G, T = RULES['refine_long_side'], RULES['global_long_side'], RULES['refine_tile']

    class TileSet(torch.utils.data.Dataset):
        def __init__(self): self.metas, self.epoch = metas, 0

        def __len__(self): return len(self.metas)

        def __getitem__(self, i):
            rng = np.random.default_rng([seed, self.epoch, i]); m = self.metas[i]; W, H = m['W'], m['H']
            st = min(L, max(W, H)) / max(W, H); sg = G / max(W, H)
            ws, hs = max(1, round(W * st)), max(1, round(H * st))
            if m['tboxes'] and rng.random() < RULES['tile_T_share']:
                _, bx0, by0, bx1, by1 = m['tboxes'][int(rng.integers(len(m['tboxes'])))]
                cx, cy = rng.uniform(bx0, bx1) * st, rng.uniform(by0, by1) * st
            else: cx, cy = rng.uniform(0, ws), rng.uniform(0, hs)
            x0 = int(np.clip(cx - T / 2, min(0, ws - T), max(0, ws - T))); y0 = int(np.clip(cy - T / 2, min(0, hs - T), max(0, hs - T)))
            ann = polygons(json.load(open(m['json'])))
            tile, tvalid = load_image(m['image'], W, H, st, (x0, y0, T, T))
            tm = rasterize_window(ann, st, x0, y0, T, T, classes=set(TIDX))[TIDX] & tvalid
            gimg, gvalid = load_image(m['image'], W, H, sg, (0, 0, G, G))
            r = sg / st; box = [x0 * r, y0 * r, (x0 + T) * r, (y0 + T) * r]
            if rng.random() < 0.5:                       # flip the tile and the image content together (padding stays)
                tile, tm, tvalid = tile[:, ::-1], tm[:, :, ::-1], tvalid[:, ::-1]
                gw = min(G, round(W * sg)); gimg = gimg.copy(); gimg[:, :gw] = gimg[:, :gw][:, ::-1]
                box = [W * sg - box[2], box[1], W * sg - box[0], box[3]]
            tile = colour_jitter(tile, np.random.default_rng([seed, self.epoch, i, 1]))
            gimg = colour_jitter(gimg, np.random.default_rng([seed, self.epoch, i, 1]))
            out = {'tile': to_tensor_img(tile), 'tmasks': torch.from_numpy(tm.copy()),
                   'tvalid': torch.from_numpy(tvalid.copy()), 'gimg': to_tensor_img(gimg),
                   'box': torch.tensor(box, dtype=torch.float32)}
            if skel in ('T', 'crack'):
                tubes = np.stack([skeleton_tubes(tm[k]) for k in range(len(TIDX))])
                out['tubes'] = torch.from_numpy(tubes)
            return out
    return TileSet()


# ---------------------------------------------------------------------------------------------------- losses
def dice_loss(p, y, v):
    """soft Dice per class over the batch (valid pixels), mean over classes; p, y (B, K, H, W), v (B, 1, H, W).
    v1.6: the same as the multilabel DiceLoss of segmentation_models_pytorch (smooth 0, eps 1e-7), which the dacl10k
    paper used: a class with no labelled pixel in the batch contributes 0. Up to v1.5 (Dice with +1 smoothing) such a
    class gave 1 - 1/(sum p + 1), which pushed its predictions to 0 everywhere."""
    dims = (0, 2, 3); p = p * v; y = y * v
    score = 2 * (p * y).sum(dims) / (p + y).sum(dims).clamp_min(1e-7)
    return ((1 - score) * (y.sum(dims) > 0).to(p.dtype)).mean()


def bce_loss(logits, y, v):
    torch, _, F = torch_mods()
    l = F.binary_cross_entropy_with_logits(logits.float(), y, reduction='none') * v
    return l.sum() / (v.sum() * logits.shape[1]).clamp(min=1)


def overlap_loss(p, y, v):
    """Dice of the predicted intersection p_a * p_b against the labelled intersection y_a * y_b for every class pair
    whose labels overlap in the image (no exclusion term for other pairs).
    v1.5: the same formula, evaluated only on the pairs of classes present in each crop (sums of elementwise
    products); v1.4 formed full 19 x 19 matrix products over all 589,824 pixels, which ran slowly on the GPU."""
    torch, _, _ = torch_mods()
    with torch.autocast(p.device.type, enabled=False):
        B, K = p.shape[:2]; pv = (p.float() * v.float()).reshape(B, K, -1); yv = (y.float() * v.float()).reshape(B, K, -1)
        pres = yv.amax(2) > 0                                          # classes labelled in each crop
        cand = pres[:, :, None] & pres[:, None, :] & torch.triu(torch.ones(K, K, dtype=torch.bool, device=p.device), 1)
        bi, ai, ci = cand.nonzero(as_tuple=True)
        if len(bi) == 0: return p.sum() * 0
        yy = (yv[bi, ai] * yv[bi, ci]).sum(1); keep = yy > 0           # pairs whose labels overlap
        if not bool(keep.any()): return p.sum() * 0
        bi, ai, ci, yy = bi[keep], ai[keep], ci[keep], yy[keep]
        pa, pc = pv[bi, ai], pv[bi, ci]
        pp = (pa * pc).sum(1); py = (pa * yv[bi, ai] * pc * yv[bi, ci]).sum(1)
        return (1 - (2 * py + 1) / (pp + yy + 1)).mean()


def overlap_loss_v14(p, y, v):
    """the v1.4 form (kept for the speed step and the equality check).
    v1.4: always 32-bit (autocast off) and divided only on the overlapping pairs. In v1.3 the pair sums (up to
    589,824 pixels of a 768 crop) ran as 16-bit matrix products under autocast and overflowed (16-bit maximum 65,504):
    the term was a constant 1 or NaN, and NaN gradients from the unused pairs stopped training."""
    torch, _, _ = torch_mods()
    with torch.autocast(p.device.type, enabled=False):
        B, K = p.shape[:2]; p = (p.float() * v.float()).reshape(B, K, -1); y = (y.float() * v.float()).reshape(B, K, -1)
        yy = y @ y.transpose(1, 2)
        present = (yy > 0) & torch.triu(torch.ones(K, K, dtype=torch.bool, device=p.device), 1)
        if not present.any(): return p.sum() * 0
        pp = p @ p.transpose(1, 2); py = (p * y) @ (p * y).transpose(1, 2)
        return (1 - (2 * py[present] + 1) / (pp[present] + yy[present] + 1)).mean()


def containment_loss(p, y, v):
    """child inside one of the parents: penalise p_child * (1 - max p_parent), skipping pixels whose labels break it."""
    ch, pa = CI[RULES['containment_child']], [CI[c] for c in RULES['containment_parents']]
    ok = 1 - (y[:, ch] * (1 - y[:, pa].amax(1)))
    w = v[:, 0] * ok
    return (p[:, ch] * (1 - p[:, pa].amax(1)) * w).sum() / w.sum().clamp(min=1)


def skeleton_recall_loss(p, tubes, v):
    """1 - soft recall of the prediction on the skeleton tubes, per class over the batch, mean over classes with tubes."""
    dims = (0, 2, 3); t = tubes * v; s = t.sum(dims)
    if (s > 0).sum() == 0: return p.sum() * 0
    rec = (p * t).sum(dims) / s.clamp(min=1)
    return (1 - rec)[s > 0].mean()


def soft_skel(x, iters=10):
    """soft skeleton of clDice (Shit et al. 2021)."""
    torch, _, F = torch_mods()

    def erode(z): return torch.min(-F.max_pool2d(-z, (3, 1), 1, (1, 0)), -F.max_pool2d(-z, (1, 3), 1, (0, 1)))

    def dilate(z): return F.max_pool2d(z, 3, 1, 1)
    sk = F.relu(x - dilate(erode(x)))
    for _ in range(iters):
        x = erode(x); d = F.relu(x - dilate(erode(x))); sk = sk + F.relu(d - sk * d)
    return sk


def cldice_loss(p, y, v):
    sp, sy = soft_skel(p * v), soft_skel(y * v)
    tprec = ((sp * y).sum() + 1) / (sp.sum() + 1); tsens = ((sy * p).sum() + 1) / (sy.sum() + 1)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def global_loss(cfg, out, b, ovl_w, cont, ovl_fn=None):
    torch, _, F = torch_mods()
    logits, pres = out[0], out[1]                                     # v1.9: the LRD also returns earlier-layer masks
    v = b['valid'].unsqueeze(1).float(); y = b['masks'].float()
    lp = RULES['loss_weights']['presence'] * F.binary_cross_entropy_with_logits(pres.float(), b['presence'])
    if cfg['head'] == 'softmax':
        lab = b['label'].masked_fill(~b['valid'], -100)
        prob = torch.softmax(logits.float(), 1)[:, 1:]
        return F.cross_entropy(logits.float(), lab, ignore_index=-100) + dice_loss(prob, y, v) + lp, {}
    p = torch.sigmoid(logits.float())
    parts = {'dice': dice_loss(p, y, v)}
    if cfg['loss'] == 'bce+dice': parts['bce'] = bce_loss(logits, y, v)
    if ovl_w > 0: parts['ovl'] = ovl_w * (ovl_fn or overlap_loss)(p, y, v)
    if cont: parts['cont'] = RULES['loss_weights']['containment'] * containment_loss(p, y, v)
    if len(out) > 2 and out[2] and cfg.get('aux_w'):                  # v1.9: deep supervision of the LRD layers
        aux = [dice_loss(torch.sigmoid(m.float()), y, v) + bce_loss(m, y, v) for m in out[2]]
        parts['aux'] = cfg['aux_w'] * sum(aux) / len(aux)
    return sum(parts.values()) + lp, {k: round(float(x.detach()), 4) for k, x in parts.items()}


def tile_loss(fused, b, skel, ovl_w):
    torch, _, _ = torch_mods()
    v = b['tvalid'].unsqueeze(1).float(); y = b['tmasks'].float(); p = torch.sigmoid(fused.float())
    parts = {'dice': dice_loss(p, y, v), 'bce': bce_loss(fused, y, v)}
    if skel == 'T': parts['skel'] = RULES['loss_weights']['skeleton_recall'] * skeleton_recall_loss(p, b['tubes'].float(), v)
    if skel == 'crack':
        k = RULES['T_classes'].index('Crack')
        parts['skel'] = RULES['loss_weights']['skeleton_recall'] * skeleton_recall_loss(
            p[:, k:k + 1], b['tubes'][:, k:k + 1].float(), v)
    if skel == 'cldice':
        k = RULES['T_classes'].index('Crack')
        parts['cldice'] = cldice_loss(p[:, k:k + 1], y[:, k:k + 1], v)
    if ovl_w > 0: parts['ovl'] = ovl_w * overlap_loss(p, y, v)
    return sum(parts.values()), {k: round(float(x.detach()), 4) for k, x in parts.items()}


# ---------------------------------------------------------------------------------------------------- prediction
def norm_pad(img, mult=32, min_hw=(0, 0)):
    """normalised tensor (1, 3, H, W) of an RGB image padded with black (as in training) to a multiple of `mult`
    and at least min_hw; returns it with the original (h, w)."""
    h, w = img.shape[:2]; H, W = max(-(-h // mult) * mult, min_hw[0]), max(-(-w // mult) * mult, min_hw[1])
    if (H, W) != (h, w):
        z = np.zeros((H, W, 3), np.uint8); z[:h, :w] = img; img = z
    return to_tensor_img(img)[None], (h, w)


def as_scores(logits, head):
    """per-class scores where > 0 means 'predicted'. Softmax: log(p_c / max over the other outputs incl. none)."""
    torch, _, _ = torch_mods()
    if head == 'sigmoid': return logits
    lp = torch.log_softmax(logits.float(), 0); out = []
    for c in range(1, lp.shape[0]):
        other = torch.cat([lp[:c], lp[c + 1:]]).amax(0); out.append(lp[c] - other)
    return torch.stack(out)


class Predictor:
    """whole-image prediction for every run type; returns parts [(class indices, scores (k, h, w), (sx, sy))]."""

    def __init__(self, cfg, model, device, glob_model=None, glob_cfg=None, tta=False):
        self.cfg, self.model, self.dev, self.g, self.gcfg, self.tta = cfg, model, device, glob_model, glob_cfg, tta

    def net(self, model, x, cfg=None):
        torch, _, _ = torch_mods()
        dt, on = amp_dtype(cfg), self.dev.type == 'cuda'
        if on and dt == torch.bfloat16 and torch.cuda.get_device_capability(self.dev)[0] < 8:
            on = False                                       # v1.8: no bfloat16 on older GPUs (T4): 32-bit instead
        with torch.autocast(self.dev.type, dtype=dt, enabled=on):
            out = model(x)
        return (out[0] if isinstance(out, tuple) else out).float()

    def whole(self, model, cfg, m):
        torch, _, F = torch_mods()
        W, H = m['W'], m['H']; sx, sy = run_scale(cfg['mode'], W, H)
        if cfg['mode'] == 'long2048':
            return self.sliding(model, cfg, m, sx)
        img = load_image(m['image'], W, H, (sx, sy))
        if not (self.tta and cfg['head'] == 'sigmoid'):
            x, (h, w) = norm_pad(img)
            return [(list(range(C)), as_scores(self.net(model, x.to(self.dev), cfg)[0, :, :h, :w], cfg['head']), (sx, sy))]
        from PIL import Image                                          # v1.7: TTA, mean probability of 3 scales x 2 flips
        hh, ww = img.shape[:2]; acc = torch.zeros((C, hh, ww), device=self.dev); n = 0
        for s in RULES['tta_scales']:
            im = img if s == 1 else np.asarray(Image.fromarray(img).resize((max(32, round(ww * s)), max(32, round(hh * s))), Image.BILINEAR))
            x, (h, w) = norm_pad(im); x = x.to(self.dev)
            for flip in (False, True):
                lo = self.net(model, torch.flip(x, [3]) if flip else x, cfg)
                lo = (torch.flip(lo, [3]) if flip else lo)[:, :, :h, :w]
                acc += torch.sigmoid(F.interpolate(lo, size=(hh, ww), mode='bilinear', align_corners=False)[0]); n += 1
        p = (acc / n).clamp(1e-6, 1 - 1e-6)
        return [(list(range(C)), torch.log(p / (1 - p)), (sx, sy))]

    def sliding(self, model, cfg, m, s):
        torch, _, _ = torch_mods()
        W, H = m['W'], m['H']; win, step = cfg['crop'], cfg['crop'] * 2 // 3
        x, (h, w) = norm_pad(load_image(m['image'], W, H, s), 32, (win, win)); x = x.to(self.dev)
        acc = torch.zeros((C, h, w), device=self.dev); cnt = torch.zeros((1, h, w), device=self.dev)
        ys = sorted({min(y, max(0, h - win)) for y in range(0, max(1, h - win + step), step)})
        xs = sorted({min(v, max(0, w - win)) for v in range(0, max(1, w - win + step), step)})
        for y0 in ys:
            for x0 in xs:
                sc = as_scores(self.net(model, x[:, :, y0:y0 + win, x0:x0 + win], cfg)[0], cfg['head'])
                ph, pw = min(win, h - y0), min(win, w - x0)
                acc[:, y0:y0 + ph, x0:x0 + pw] += sc[:, :ph, :pw]; cnt[:, y0:y0 + ph, x0:x0 + pw] += 1
        return [(list(range(C)), acc / cnt.clamp(min=1), (s, s))]

    def refine(self, m):
        """global path at long side 1024 for all classes; refinement tiles at long side min(2048, native) for T where
        the global probability of any T class exceeds 0.1."""
        torch, _, F = torch_mods()
        from torchvision.ops import roi_align
        W, H = m['W'], m['H']; parts = self.whole(self.g, self.gcfg, m); gs = parts[0][1]; sg = parts[0][2][0]
        st = min(RULES['refine_long_side'], max(W, H)) / max(W, H); T = RULES['refine_tile']
        x, (h, w) = norm_pad(load_image(m['image'], W, H, st), T); x = x.to(self.dev); r = sg / st
        canvas = F.interpolate(gs[TIDX][None], size=(h, w), mode='bilinear', align_corners=False)[0]
        prob_T = torch.sigmoid(gs[TIDX]).amax(0)
        tiles = []
        for y0 in range(0, h, T):
            for x0 in range(0, w, T):
                gy0, gx0, gy1, gx1 = int(y0 * r), int(x0 * r), int(np.ceil((y0 + T) * r)), int(np.ceil((x0 + T) * r))
                if prob_T[gy0:gy1, gx0:gx1].numel() and prob_T[gy0:gy1, gx0:gx1].max() > RULES['refine_select_prob']:
                    tiles.append((x0, y0))
        self.last_tiles = len(tiles)
        for k in range(0, len(tiles), 8):
            batch = tiles[k:k + 8]
            t = torch.cat([x[:, :, y0:y0 + T, x0:x0 + T] for x0, y0 in batch])
            boxes = torch.tensor([[0, x0 * r, y0 * r, (x0 + T) * r, (y0 + T) * r] for x0, y0 in batch],
                                 dtype=torch.float32, device=self.dev)                 # all boxes on the one image
            ctx = roi_align(gs[None].float(), boxes, (T // 2, T // 2), 1.0, 2, True)
            with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == 'cuda'):
                fused = self.model(t, ctx).float()
            for i, (x0, y0) in enumerate(batch):
                ph, pw = min(T, h - y0), min(T, w - x0); canvas[:, y0:y0 + ph, x0:x0 + pw] = fused[i, :, :ph, :pw]
        nonT = [c for c in range(C) if c not in TIDX]
        return [(nonT, gs[nonT], (sg, sg)), (TIDX, canvas, (st, st))]

    def __call__(self, m):
        torch, _, _ = torch_mods()
        with torch.no_grad():
            return self.refine(m) if self.cfg.get('stage') == 2 else self.whole(self.model, self.cfg, m)


def resample(parts, W, H, sx, sy, dev):
    """scores of all 19 classes at the resolution (round(W * sx), round(H * sy))."""
    torch, _, F = torch_mods()
    h, w = max(1, round(H * sy)), max(1, round(W * sx)); out = torch.empty((C, h, w), device=dev, dtype=torch.float16)
    for idx, sc, _ in parts:
        for k, c in enumerate(idx):
            out[c] = F.interpolate(sc[k][None, None].float(), size=(h, w), mode='bilinear', align_corners=False)[0, 0].half()
    return out


# ---------------------------------------------------------------------------------------------------- evaluation
def toolkit_masks(gn):
    """labels as built by the dacl10k toolkit (Dacl10kDataset): full-size uint8 masks resized to 512 x 512 with
    skimage.transform.resize (anti-aliasing, bilinear), then every positive value counts. One class at a time."""
    from skimage.transform import resize
    out = np.zeros((C, 512, 512), bool)
    for c in np.flatnonzero(gn.reshape(C, -1).any(1).cpu().numpy()):
        out[c] = resize(gn[c].cpu().numpy().astype(np.uint8), (512, 512)) > 0
    return out


def image_metrics(parts, m, dev, full=True, toolkit=False, thr=None):
    """per-image counts for every S1 metric (summed over images later). full=False: only IoU at long side 1024.
    toolkit=True adds the toolkit-protocol IoU (labels thickened as in the dacl10k toolkit, at 512 x 512).
    v1.8: thr = per-class thresholds as logits (19 values); None = probability 0.5 for every class."""
    torch, _, _ = torch_mods()
    t = 0 if thr is None else torch.as_tensor(thr, dtype=torch.float32, device=dev).view(C, 1, 1)
    from scipy.ndimage import distance_transform_edt, label
    W, H = m['W'], m['H']; polys = polygons(json.load(open(m['json']))); rec = {}

    def iu(pred, gt):
        return np.stack([(pred & gt).sum((1, 2)).cpu().numpy(), (pred | gt).sum((1, 2)).cpu().numpy(),
                         pred.sum((1, 2)).cpu().numpy(), gt.sum((1, 2)).cpu().numpy()], 1)
    s = RULES['global_long_side'] / max(W, H)
    p1024 = resample(parts, W, H, s, s, dev) > t
    g1024 = torch.from_numpy(rasterize_window(polys, s, 0, 0, p1024.shape[2], p1024.shape[1])).to(dev)
    rec['iu1024'] = iu(p1024, g1024)
    if not full: return rec, p1024
    pn = resample(parts, W, H, 1.0, 1.0, dev) > t                          # native resolution
    gn = torch.from_numpy(rasterize_window(polys, 1.0, 0, 0, W, H)).to(dev)
    rec['iu_native'] = iu(pn, gn)
    multi = gn.sum(0) >= 2
    rec['multi'] = np.array([int(multi.sum()), int(((pn == gn).all(0) & multi).sum())])
    pairs = [('ExposedRebars', 'Rust'), ('ExposedRebars', 'Spalling'), ('Spalling', 'Rust')]
    rec['pairs'] = np.array([[int(((pn[CI[a]] & pn[CI[b]]) & (gn[CI[a]] & gn[CI[b]])).sum()),
                              int(((pn[CI[a]] & pn[CI[b]]) | (gn[CI[a]] & gn[CI[b]])).sum())] for a, b in pairs])
    gtk = toolkit_masks(gn) if toolkit else None
    del pn, gn, multi
    p512 = resample(parts, W, H, 512 / W, 512 / H, dev) > t
    g512 = torch.from_numpy(rasterize_window(polys, (512 / W, 512 / H), 0, 0, 512, 512)).to(dev)
    rec['iu512'] = iu(p512, g512)
    if toolkit:
        gtk = torch.from_numpy(gtk).to(dev)
        rec['iu512_toolkit'] = iu(p512, gtk)
        rec['toolkit_area_ratio'] = np.stack([gtk.sum((1, 2)).cpu().numpy(), g512.sum((1, 2)).cpu().numpy()], 1)
    # thin and small damage at long side 2048
    s2 = RULES['topology_eval_long_side'] / max(W, H); h2, w2 = max(1, round(H * s2)), max(1, round(W * s2))
    p2 = (resample(parts, W, H, s2, s2, dev) > t).cpu().numpy()
    want = [CI[c] for c in ('Crack', 'ACrack', 'Cavity', 'Rust', 'ExposedRebars')]
    g2 = rasterize_window(polys, s2, 0, 0, w2, h2, classes=set(want))
    cl = []
    for c in ('Crack', 'ACrack'):
        P, G = p2[CI[c]], g2[CI[c]]; sp, sg = skeleton(P), skeleton(G)
        cl.append([int((sp & G).sum()), int(sp.sum()), int((sg & P).sum()), int(sg.sum())])
    rec['cldice'] = np.array(cl)
    P, G = p2[CI['Crack']], g2[CI['Crack']]
    rec['b0'] = np.array([label(P, np.ones((3, 3)))[1], label(G, np.ones((3, 3)))[1]])
    sp, sg = skeleton(P), skeleton(G); tol = RULES['centerline_tolerance_px']
    dG = distance_transform_edt(~sg) if sg.any() else np.full(P.shape, np.inf)
    dP = distance_transform_edt(~sp) if sp.any() else np.full(P.shape, np.inf)
    rec['centerline'] = np.array([int((sp & (dG <= tol)).sum()), int(sp.sum()), int((sg & (dP <= tol)).sum()), int(sg.sum())])
    inst = []
    for c in ('Cavity', 'Rust', 'ExposedRebars'):
        found = total = 0
        for k, q in polys:
            if k != CI[c]: continue
            x0, y0 = np.floor(q.min(0) * s2).astype(int); x1, y1 = np.ceil(q.max(0) * s2).astype(int) + 1
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w2, x1), min(h2, y1)
            if x1 <= x0 or y1 <= y0: continue
            mk = rasterize_window([(k, q)], s2, x0, y0, x1 - x0, y1 - y0, classes={k})[k]
            if not mk.any(): continue
            total += 1; found += int(p2[k, y0:y1, x0:x1][mk].mean() >= RULES['instance_hit_share'])
        inst.append([found, total])
    rec['instance'] = np.array(inst)
    return rec, p1024


def summarise(recs):
    """dataset-level metrics from the per-image counts."""
    def iou(key):
        a = np.sum([r[key] for r in recs], 0).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'): v = a[:, 0] / a[:, 1]
        return v
    out = {}
    for key in ('iu_native', 'iu512', 'iu512_toolkit', 'iu1024'):
        if key not in recs[0]: continue
        v = iou(key); out[key] = {'per_class': {CLASSES[c]: (None if np.isnan(v[c]) else round(float(v[c]), 4)) for c in range(C)},
                                  'mIoU': round(float(np.nanmean(v)), 4),
                                  'T_mIoU': round(float(np.nanmean(v[TIDX])), 4),
                                  'rare_mIoU': round(float(np.nanmean(v[[CI[c] for c in RULES['rare_classes']]])), 4)}
    if 'toolkit_area_ratio' in recs[0]:                 # label area of toolkit masks / exact masks, per class
        ar = np.sum([r['toolkit_area_ratio'] for r in recs], 0).astype(float)
        out['toolkit_label_area_ratio'] = {CLASSES[c]: (round(float(ar[c, 0] / ar[c, 1]), 3) if ar[c, 1] else None)
                                           for c in range(C)}
    if 'cldice' in recs[0]:
        cd = np.sum([r['cldice'] for r in recs], 0).astype(float); cdv = {}
        for i, c in enumerate(('Crack', 'ACrack')):
            tp, ts = cd[i, 0] / max(cd[i, 1], 1), cd[i, 2] / max(cd[i, 3], 1)
            cdv[c] = round(2 * tp * ts / (tp + ts), 4) if tp + ts > 0 else 0.0
        out['clDice'] = cdv
        b0 = np.array([r['b0'] for r in recs]); has = (b0 > 0).any(1)
        out['crack_components_abs_error'] = round(float(np.abs(b0[has, 0] - b0[has, 1]).mean()), 3) if has.any() else None
        cc = np.sum([r['centerline'] for r in recs], 0).astype(float)
        pr, rc = cc[0] / max(cc[1], 1), cc[2] / max(cc[3], 1)
        out['crack_centerline'] = {'precision': round(pr, 4), 'recall': round(rc, 4),
                                   'F1': round(2 * pr * rc / (pr + rc), 4) if pr + rc > 0 else 0.0}
        ins = np.sum([r['instance'] for r in recs], 0)
        out['instance_recall'] = {c: round(float(ins[i, 0] / max(ins[i, 1], 1)), 4) for i, c in enumerate(('Cavity', 'Rust', 'ExposedRebars'))}
        mu = np.sum([r['multi'] for r in recs], 0)
        out['multilabel_exact_match'] = round(float(mu[1] / max(mu[0], 1)), 4)
        pa = np.sum([r['pairs'] for r in recs], 0)
        out['pair_intersection_IoU'] = {n: round(float(pa[i, 0] / max(pa[i, 1], 1)), 4) for i, n in
                                        enumerate(('ExposedRebars&Rust', 'ExposedRebars&Spalling', 'Spalling&Rust'))}
    return out


def evaluate(pred, metas, dev, out_dir, tag, full=True, save_preds=True, chunk=100, toolkit=None):
    """metrics for every image (resumable in chunks of `chunk` images, saved on Drive) and, when save_preds, the
    predicted masks at long side 1024 (bit-packed) and the probabilities at long side 512 (uint8) of every image."""
    torch, _, F = torch_mods()
    os.makedirs(os.path.join(out_dir, f'eval_{tag}'), exist_ok=True); recs, names, t_img, tiles = [], [], [], []
    for c0 in range(0, len(metas), chunk):
        cpath = os.path.join(out_dir, f'eval_{tag}', f'chunk_{c0:05d}.npz')
        if os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=True); recs += list(z['recs']); names += list(z['names']); continue
        crecs, cnames, buf = [], [], {}
        for m in metas[c0:c0 + chunk]:
            if dev.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time(); parts = pred(m)
            if dev.type == 'cuda': torch.cuda.synchronize()
            t_img.append(time.time() - t0); tiles.append(getattr(pred, 'last_tiles', 0))
            tk = (full and tag.startswith('validation')) if toolkit is None else toolkit
            r, p1024 = image_metrics(parts, m, dev, full, tk); crecs.append(r); cnames.append(m['name'])
            if save_preds:
                W, H = m['W'], m['H']; s5 = 512 / max(W, H)
                prob = torch.sigmoid(resample(parts, W, H, s5, s5, dev).float())
                buf[m['name']] = {'mask1024': np.packbits(p1024.cpu().numpy(), axis=-1), 'shape1024': np.array(p1024.shape),
                                  'prob512': (prob * 255).round().byte().cpu().numpy()}
        if save_preds:
            with zipfile.ZipFile(os.path.join(out_dir, f'eval_{tag}', f'preds_{c0:05d}.zip.part'), 'w', zipfile.ZIP_DEFLATED) as z:
                for n, d in buf.items():
                    b = io.BytesIO(); np.savez(b, **d); z.writestr(os.path.splitext(n)[0] + '.npz', b.getvalue())
            os.replace(os.path.join(out_dir, f'eval_{tag}', f'preds_{c0:05d}.zip.part'),
                       os.path.join(out_dir, f'eval_{tag}', f'preds_{c0:05d}.zip'))
        np.savez(cpath + '.part.npz', recs=np.array(crecs, dtype=object), names=np.array(cnames))
        os.replace(cpath + '.part.npz', cpath); recs += crecs; names += cnames
        log(f'  {tag}: {len(recs)} / {len(metas)} images evaluated')
    res = summarise(recs)
    if t_img: res['seconds_per_image'] = round(float(np.mean(t_img)), 3); res['refined_tiles_per_image'] = round(float(np.mean(tiles)), 2)
    return res, recs, names


def example_figure_run(pred, metas, dev, path, n=6):
    """GT and prediction of the thin/small classes for a fixed set of validation images (same for every run)."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt, matplotlib.colors, matplotlib.patches
    torch, _, _ = torch_mods()
    cols = {'Crack': '#d62728', 'Cavity': '#ff7f0e', 'ExposedRebars': '#9467bd', 'Rust': '#1f77b4'}
    pick = [m for m in sorted(metas, key=lambda m: m['name']) if sum(m['presence'][CI[c]] for c in cols) >= 2][:n]
    if not pick: return
    fig, axs = plt.subplots(len(pick), 3, figsize=(10.5, 2.7 * len(pick)), squeeze=False)
    for r, m in enumerate(pick):
        W, H = m['W'], m['H']; s = RULES['global_long_side'] / max(W, H)
        img = load_image(m['image'], W, H, s); h, w = img.shape[:2]
        gt = rasterize_window(polygons(json.load(open(m['json']))), s, 0, 0, w, h)
        pr = (resample(pred(m), W, H, s, s, dev) > 0).cpu().numpy()
        for ax, ms, title in ((axs[r, 0], None, m['name']), (axs[r, 1], gt, 'labels'), (axs[r, 2], pr, 'prediction')):
            ax.imshow(img)
            if ms is not None:
                over = np.zeros((h, w, 4))
                for c, col in cols.items(): over[ms[CI[c]]] = (*matplotlib.colors.to_rgb(col), 0.55)
                ax.imshow(over)
            ax.set_title(title, fontsize=7, loc='left'); ax.axis('off')
    fig.legend(handles=[matplotlib.patches.Patch(color=v, label=k) for k, v in cols.items()], loc='lower center',
               ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


def count_gflops(model, cfg, dev):
    """GFLOPs of one forward pass at the run's input size (B3-HR: one 768 x 768 window; Full: one 512 tile)."""
    torch, _, _ = torch_mods()
    from torch.utils.flop_counter import FlopCounterMode
    hw = {'fixed512': (512, 512), 'long1024': (768, 1024), 'long2048': (768, 768), 'tile': (512, 512)}[cfg['mode']]
    x = torch.zeros(1, 3, *hw, device=dev)
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x, torch.zeros(1, C, hw[0] // 2, hw[1] // 2, device=dev)) if cfg.get('stage') == 2 else model(x)
        return {'GFLOPs': round(fc.get_total_flops() / 1e9, 1), 'input': list(hw)}
    except Exception as e:                                                   # never stop a run for this
        return {'GFLOPs': None, 'error': repr(e)[:200]}


# ---------------------------------------------------------------------------------------------------- training loop
def load_index(a):
    p, q = os.path.join(a.out, 's2_index.json'), os.path.join(a.out, 's2_split.json')
    if not (os.path.exists(p) and os.path.exists(q)): sys.exit('run step s2 first (index and split)')
    idx = json.load(open(p))['index']; sp = json.load(open(q))
    fit, ho = set(sp['fit']), set(sp['holdout'])
    tr = [m for m in idx if m['split'] == 'train']
    return [m for m in tr if m['name'] in fit], [m for m in tr if m['name'] in ho], [m for m in idx if m['split'] == 'validation']


def relink(metas, root):
    """index paths point to the session where s2 ran; map them to this session's unpacked folder."""
    out, dirs = [], {}
    for m in metas:
        m = dict(m); split = m['split']
        if split not in dirs: dirs[split] = (find_split_dir(root, 'annotations', split), find_split_dir(root, 'images', split))
        m['json'] = os.path.join(dirs[split][0], os.path.basename(m['json']))
        m['image'] = os.path.join(dirs[split][1], os.path.basename(m['image'])); out.append(m)
    return out


def containment_used(a):
    p = os.path.join(a.out, 's0b_stats.json')
    if not os.path.exists(p): sys.exit('run step s0b first (it decides the containment term)')
    return bool(json.load(open(p))['containment_term_used'])


def save_ckpt(obj, path, drive_path=None):
    torch, _, _ = torch_mods()
    torch.save(obj, path + '.part'); os.replace(path + '.part', path)
    if drive_path:
        try: shutil.copy(path, drive_path + '.part'); os.replace(drive_path + '.part', drive_path)
        except OSError as e: log(f'  WARNING: could not write {drive_path} to Drive ({e}); the local copy is kept')


OVERLAP_FP16_BUG = ('paperc.py', 'paperc_v1_1.py', 'paperc_v1_2.py', 'paperc_v1_3.py')   # overlap term overflowed (v1.4)
OLD_DICE = OVERLAP_FP16_BUG + ('paperc_v1_4.py', 'paperc_v1_5.py')                       # Dice loss before v1.6


def finite_state(sd):
    """True if every floating-point tensor of a state dict (weights and BatchNorm statistics) is finite."""
    torch, _, _ = torch_mods()
    return all(bool(torch.isfinite(t).all()) for t in sd.values() if torch.is_tensor(t) and t.is_floating_point())


def move_aside(dirs, tag):
    """keep an invalid earlier run for the record under a new folder name, so the same command starts it again."""
    for d in dirs:
        if os.path.isdir(d):
            new, k = d + tag, 1
            while os.path.exists(new): k += 1; new = f'{d}{tag}-{k}'
            shutil.move(d, new); print(f'kept the earlier run as {new}', flush=True)


def load_stage1(a, stage1, dev):
    """frozen global path of a finished sigmoid run (B3 or B3-ovl) for the refinement stage."""
    torch, _, _ = torch_mods()
    sdir = os.path.join(a.out, 'runs', stage1)
    if not os.path.exists(os.path.join(sdir, 'done.json')): sys.exit(f'stage 1 run {stage1} is not finished yet')
    if json.load(open(os.path.join(sdir, 'done.json'))).get('screening'): sys.exit('a screening run cannot be stage 1')
    scfg = json.load(open(os.path.join(sdir, 'config.json')))
    if scfg.get('code') in OLD_DICE:
        sys.exit(f'stage 1 run {stage1} was trained with the Dice loss of {scfg["code"]}; run it again with {VERSION} first')
    if scfg.get('overlap_weight') and scfg.get('code') in OVERLAP_FP16_BUG:
        sys.exit(f'stage 1 run {stage1} was made by {scfg["code"]} (overlap term NaN); run it again with {VERSION}')
    if CONFIGS[scfg['model']]['head'] != 'sigmoid' or CONFIGS[scfg['model']].get('stage'):
        sys.exit('stage 1 must be a sigmoid global path (B3 or B3-ovl)')
    gcfg = dict(CONFIGS[scfg['model']]); gmodel = build_model(gcfg, pretrained=False, small=a.small).to(dev)
    gmodel.load_state_dict({k: v.float() for k, v in torch.load(os.path.join(sdir, 'best.pt'), map_location=dev)['model'].items()})
    gmodel.eval(); [p.requires_grad_(False) for p in gmodel.parameters()]
    return gmodel, gcfg


def step_speed(a, root):
    """about 3-5 minutes on the GPU, nothing saved. For the model given with --model (default B3-ovl): (1) the data
    loader alone, (2) one training step (8 images) with the baseline loss and with our loss terms, peak GPU memory,
    (3) whole-image prediction time, and the resulting time per epoch and per run. v1.7: any configuration, including
    the large ViT; it also shows whether the pretrained weights download."""
    ensure_smp()
    torch, nn, F = torch_mods(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fit, ho, _ = (relink(x, root) for x in load_index(a))
    mname = a.model or 'B3-ovl'; cfg = dict(CONFIGS[mname]); cfg['ovl'] = True; B = cfg['batch']
    log(f'model {mname} ({cfg["arch"]}), GPU {torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"}, '
        f'torch {torch.__version__}, workers {a.workers}, batch {B} x accumulation {cfg["accum"]}')
    data = make_train_set(fit, cfg, 0)
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(0)).tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, order[:B * 43]), batch_size=B, num_workers=a.workers,
                                         pin_memory=dev.type == 'cuda', drop_last=True)
    t = None; n = 0; first = None
    for i, b in enumerate(loader):
        if i == 0: first = b
        if i == 3: t = time.time()
        if i >= 3: n += B
    lps = n / max(1e-6, time.time() - t) if t else float('nan')
    log(f'1. data loader alone: {lps:.1f} images/s, i.e. {8 / lps:.2f} s per training step of 8 images at best')
    t0 = time.time(); model = build_model(cfg, pretrained=not a.small, small=a.small).to(dev); model.train()
    log(f'   model built in {time.time() - t0:.0f} s (includes the weight download); '
        f'{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters')
    if a.grad_ckpt and hasattr(model.encoder, 'set_grad_checkpointing'): model.encoder.set_grad_checkpointing(True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda' and cfg.get('amp') != 'bf16')
    b = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in first.items()}; rows = []
    w = float(a.ovl_weight) if a.ovl_weight else 0.3
    for label, ovl_w, cont in (('baseline loss (BCE + Dice)', 0.0, False), (f'our loss (+ overlap {w:g} + containment)', w, True)):
        if dev.type == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for it in range(13):
            if it == 3:
                if dev.type == 'cuda': torch.cuda.synchronize()
                t = time.time()
            for _ in range(cfg['accum']):
                with torch.autocast(dev.type, dtype=amp_dtype(cfg), enabled=dev.type == 'cuda'): out = model(b['img'])
                loss, _ = global_loss(cfg, out, b, ovl_w, cont); scaler.scale(loss / cfg['accum']).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if dev.type == 'cuda': torch.cuda.synchronize()
        rows.append((label, (time.time() - t) / 10, torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else 0))
    model.eval(); pred = Predictor(cfg, model, dev); k = min(10, len(ho))
    pred(ho[0]); t = time.time()
    for m in ho[1:k + 1]: pred(m)
    per_img = (time.time() - t) / max(1, k)
    steps = len(fit) // (B * cfg['accum'])
    log('2. one training step (8 images):')
    for label, sec, mem in rows:
        ep = max(sec, 8 / lps) * steps / 60 + per_img * len(ho) / 60
        log(f'   {label:38s} {sec:.2f} s/step, peak {mem:.1f} GB -> about {ep:.1f} min per epoch incl. hold-out check, '
            f'{ep * 20 / 60:.1f} h for 20 epochs')
    log(f'3. prediction: {per_img:.2f} s per image (hold-out check {per_img * len(ho) / 60:.1f} min per epoch)')
    log('Please paste these lines into the chat.')


def step_eval(a, root):
    """re-evaluate a finished run from its stored best checkpoint, without training, with the metrics of this code
    version (adds the toolkit-protocol score). Writes results_<split>_<version>.json, per_image_<split>_<version>.npz and
    log.txt lines; earlier result files are kept. Resumable in blocks of 100 images."""
    ensure_smp()
    torch, _, _ = torch_mods()
    if not a.run_name: sys.exit('give --run <run name>, e.g. B1-s0')
    rdir = os.path.join(a.out, 'runs', a.run_name)
    if not os.path.exists(os.path.join(rdir, 'best.pt')): sys.exit(f'no best.pt in {rdir}')
    global LOG_PATH; LOG_PATH = os.path.join(rdir, 'log.txt'); log(f'--- eval {VERSION} | {time.strftime("%Y-%m-%d %H:%M")}')
    meta = json.load(open(os.path.join(rdir, 'config.json'))); cfg = dict(CONFIGS[meta['model']])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, ho, val = (relink(x, root) for x in load_index(a))
    if a.max_images: ho, val = ho[:max(2, a.max_images // 4)], val[:max(2, a.max_images // 4)]
    gmodel = gcfg = None
    if cfg.get('stage') == 2:
        gmodel, gcfg = load_stage1(a, meta['stage1'], dev); model = make_refine_head().to(dev)
    else: model = build_model(cfg, pretrained=False, small=a.small).to(dev)
    best = torch.load(os.path.join(rdir, 'best.pt'), map_location=dev)
    model.load_state_dict({k: v.float() for k, v in best['model'].items()}); model.eval()
    pred = Predictor(cfg, model, dev, gmodel, gcfg, tta=a.tta); v = VERSION.replace('paperc_', '').replace('.py', '')
    if a.tta: v += '_tta'; log('test-time augmentation: scales ' + ', '.join(f'{s:g}' for s in RULES['tta_scales']) + ' and flip')
    out = {'run': a.run_name, 'best_epoch': best['epoch'], 'code': VERSION}
    for tag, metas in (('validation', val), ('holdout', ho)):
        res, recs, names = evaluate(pred, metas, dev, rdir, f'{tag}_{v}', full=True, save_preds=False,
                                    toolkit=(tag == 'validation'))
        out[tag] = res; save_json(res, os.path.join(rdir, f'results_{tag}_{v}.json'))
        np.savez(os.path.join(rdir, f'per_image_{tag}_{v}.npz'), names=np.array(names), **{
            k: np.stack([r[k] for r in recs]) for k in recs[0]})
        msg = f'{tag}: mIoU native {res["iu_native"]["mIoU"]:.4f}, 512x512 {res["iu512"]["mIoU"]:.4f}'
        if 'iu512_toolkit' in res: msg += f', toolkit protocol {res["iu512_toolkit"]["mIoU"]:.4f}'
        log(msg)
    save_json(out, os.path.join(rdir, f'eval_{v}.json'))
    if 'iu512_toolkit' in out['validation']:
        log('\nper class (validation): native | 512x512 | toolkit protocol | toolkit label area / exact label area')
        r = out['validation']
        for c in CLASSES:
            log(f'  {c:15s} {r["iu_native"]["per_class"][c]} | {r["iu512"]["per_class"][c]} | '
                f'{r["iu512_toolkit"]["per_class"][c]} | {r["toolkit_label_area_ratio"][c]}')
    log(f'written eval_{v}.json, results_*_{v}.json and per_image_*_{v}.npz in {rdir}')


# ---------------------------------------------------------------------------------------------------- v1.8: final step
def average_epochs(rdir, epochs):
    """mean of the stored 16-bit weights of the given epochs, computed in 32-bit on the CPU."""
    torch, _, _ = torch_mods()
    acc = None
    for e in epochs:
        sd = torch.load(os.path.join(rdir, f'epoch_{e:03d}.pt'), map_location='cpu')['model']
        if acc is None: acc = {k: v.float().clone() for k, v in sd.items()}
        else:
            for k, v in sd.items(): acc[k] += v.float()
        del sd
    return {k: v / len(epochs) for k, v in acc.items()}


def threshold_hist(pred, metas, dev, d, chunk=100):
    """per image and class: counts of labelled and unlabelled pixels in 100 probability bins, at long side 1024
    (the selection resolution). Resumable in chunks; returns an array (images, 2, C, bins)."""
    torch, _, _ = torch_mods()
    NB = RULES['final_threshold_bins']; os.makedirs(d, exist_ok=True); out, t_img = [], []
    for c0 in range(0, len(metas), chunk):
        cpath = os.path.join(d, f'hist_{c0:05d}.npz')
        if os.path.exists(cpath): out.append(np.load(cpath)['hist']); continue
        hs = []
        for m in metas[c0:c0 + chunk]:
            t0 = time.time(); parts = pred(m); W, H = m['W'], m['H']; s = RULES['global_long_side'] / max(W, H)
            p = torch.sigmoid(resample(parts, W, H, s, s, dev).float())
            g = torch.from_numpy(rasterize_window(polygons(json.load(open(m['json']))), s, 0, 0, p.shape[2], p.shape[1])).to(dev)
            idx = (p * NB).long().clamp_(0, NB - 1) + torch.arange(C, device=dev).view(C, 1, 1) * NB
            hs.append(np.stack([torch.bincount(idx[g], minlength=C * NB).view(C, NB).cpu().numpy(),
                                torch.bincount(idx[~g], minlength=C * NB).view(C, NB).cpu().numpy()]))
            t_img.append(time.time() - t0)
        h = np.stack(hs).astype(np.int64); np.savez_compressed(cpath + '.part.npz', hist=h); os.replace(cpath + '.part.npz', cpath)
        out.append(h); log(f'  hold-out with TTA: {sum(len(x) for x in out)} / {len(metas)} images'
                           + (f', {np.mean(t_img):.2f} s per image' if t_img else ''))
    return np.concatenate(out)


def hist_iou(h, k):
    """per-class IoU from summed histograms h (2, C, bins) when bins >= k[c] count as predicted (nan if no union)."""
    pos, neg = h[0].astype(float), h[1].astype(float)
    tp = np.array([pos[c, k[c]:].sum() for c in range(C)]); fp = np.array([neg[c, k[c]:].sum() for c in range(C)])
    u = pos.sum(1) + fp
    with np.errstate(invalid='ignore', divide='ignore'): return np.where(u > 0, tp / u, np.nan)


def tune_thresholds(h):
    """per class, the grid threshold with the highest IoU on h (ties: the one closest to 0.5); 0.5 if no labels."""
    NB = RULES['final_threshold_bins']; grid = RULES['final_threshold_grid']; k = np.full(C, NB // 2)
    ious = np.stack([hist_iou(h, np.full(C, round(t * NB))) for t in grid])          # (grid, C)
    for c in range(C):
        if h[0, c].sum() == 0 or np.all(np.isnan(ious[:, c])): continue
        best = np.nanmax(ious[:, c]); j = min((i for i in range(len(grid)) if ious[i, c] >= best - 1e-12),
                                               key=lambda i: abs(grid[i] - 0.5))
        k[c] = round(grid[j] * NB)
    return k


def evaluate_final(pred, metas, dev, d, thr_sets, chunk=100):
    """every S1 metric for every image, once per threshold set (one prediction per image); saves the masks of each set
    at long side 1024 (bit-packed) and the probabilities at long side 512 (uint8) of every image. Resumable."""
    torch, _, _ = torch_mods()
    os.makedirs(d, exist_ok=True); recs = {k: [] for k in thr_sets}; names, t_img = [], []
    for c0 in range(0, len(metas), chunk):
        cpath = os.path.join(d, f'chunk_{c0:05d}.npz')
        if os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=True); names += list(z['names'])
            for k in thr_sets: recs[k] += list(z['recs_' + k])
            continue
        crecs, cnames, buf = {k: [] for k in thr_sets}, [], {}
        for m in metas[c0:c0 + chunk]:
            if dev.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time(); parts = pred(m)
            if dev.type == 'cuda': torch.cuda.synchronize()
            t_img.append(time.time() - t0); W, H = m['W'], m['H']; s5 = 512 / max(W, H)
            item = {'prob512': (torch.sigmoid(resample(parts, W, H, s5, s5, dev).float()) * 255).round().byte().cpu().numpy()}
            for k, thr in thr_sets.items():
                r, p1024 = image_metrics(parts, m, dev, True, True, thr); crecs[k].append(r)
                item['mask1024_' + k] = np.packbits(p1024.cpu().numpy(), axis=-1); item['shape1024'] = np.array(p1024.shape)
            cnames.append(m['name']); buf[m['name']] = item
        zp = os.path.join(d, f'preds_{c0:05d}.zip')
        with zipfile.ZipFile(zp + '.part', 'w', zipfile.ZIP_DEFLATED) as z:
            for n, it in buf.items():
                b = io.BytesIO(); np.savez(b, **it); z.writestr(os.path.splitext(n)[0] + '.npz', b.getvalue())
        os.replace(zp + '.part', zp)
        np.savez(cpath + '.part.npz', names=np.array(cnames), **{'recs_' + k: np.array(v, dtype=object) for k, v in crecs.items()})
        os.replace(cpath + '.part.npz', cpath); names += cnames
        for k in thr_sets: recs[k] += crecs[k]
        log(f'  validation: {len(names)} / {len(metas)} images' + (f', {np.mean(t_img):.2f} s per image (prediction)' if t_img else ''))
    return {k: summarise(v) for k, v in recs.items()}, recs, names


def step_final(a, root):
    """v1.8: final result of a finished stage-1 run without training (see the module text and RULES 'changes').
    Every choice uses the hold-out part only; the validation split is evaluated once, at the end. Resumable."""
    ensure_smp()
    torch, _, _ = torch_mods()
    if not a.run_name: sys.exit('give --run <run name>, e.g. B3-ovl-L0.1-s0')
    rdir = os.path.join(a.out, 'runs', a.run_name)
    if not os.path.exists(os.path.join(rdir, 'done.json')): sys.exit(f'{a.run_name} is not finished yet (no done.json in {rdir})')
    if json.load(open(os.path.join(rdir, 'done.json'))).get('screening'): sys.exit('final: not for screening runs')
    meta = json.load(open(os.path.join(rdir, 'config.json'))); cfg = dict(CONFIGS[meta['model']])
    if cfg.get('stage') == 2 or cfg['head'] != 'sigmoid': sys.exit('final (v1.8): sigmoid stage-1 runs only (B3, B3-ovl, B3-L, B3-ovl-L)')
    if meta.get('code') in OLD_DICE: sys.exit(f'{a.run_name} was trained with the Dice loss of {meta["code"]}')
    global LOG_PATH; LOG_PATH = os.path.join(rdir, 'log.txt'); log(f'--- final {VERSION} | {time.strftime("%Y-%m-%d %H:%M")}')
    fdir = os.path.join(rdir, 'final_v1_8'); os.makedirs(fdir, exist_ok=True); spath = os.path.join(fdir, 'state.json')
    st = json.load(open(spath)) if os.path.exists(spath) else {}
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'GPU {torch.cuda.get_device_name(0) if dev.type == "cuda" else "none (CPU)"}')
    _, ho, val = (relink(x, root) for x in load_index(a))
    if a.max_images: ho, val = ho[:max(4, a.max_images // 4)], val[:max(2, a.max_images // 4)]
    model = build_model(cfg, pretrained=False, small=a.small).to(dev).eval()
    best = torch.load(os.path.join(rdir, 'best.pt'), map_location='cpu')
    hist = json.load(open(os.path.join(rdir, 'history.json'))); E = len(hist)

    # 1. weights, chosen on the hold-out part (mIoU at long side 1024, threshold 0.5, no TTA)
    cands = st.get('candidates', {})
    if 'best' not in cands:
        mb = best.get('holdout_mIoU_1024')
        if mb is None:                                             # not stored (hold-out not checked that epoch)
            model.load_state_dict({k: v.float() for k, v in best['model'].items()})
            mb = evaluate(Predictor(cfg, model, dev), ho, dev, fdir, 'holdout_best', full=False, save_preds=False)[0]['iu1024']['mIoU']
        cands['best'] = {'epochs': [best['epoch']], 'holdout_mIoU_1024': mb}
    for cname, K in RULES['final_weight_candidates'].items():
        if K == 0 or K > E or cname in cands: continue
        eps = list(range(E - K + 1, E + 1))
        if not all(os.path.exists(os.path.join(rdir, f'epoch_{e:03d}.pt')) for e in eps):
            log(f'  {cname}: epoch files missing, skipped'); continue
        model.load_state_dict(average_epochs(rdir, eps)); model.eval()
        res, _, _ = evaluate(Predictor(cfg, model, dev), ho, dev, fdir, f'holdout_{cname}', full=False, save_preds=False)
        cands[cname] = {'epochs': eps, 'holdout_mIoU_1024': res['iu1024']['mIoU'], 'holdout_T_mIoU_1024': res['iu1024']['T_mIoU']}
        st['candidates'] = cands; save_json(st, spath)
    log('1. weights (hold-out mIoU at long side 1024, threshold 0.5, no TTA):')
    for cname, v in cands.items():
        ep = v['epochs']; log(f'   {cname:6s} epoch{"s" if len(ep) > 1 else ""} {ep[0]}' + (f'-{ep[-1]}' if len(ep) > 1 else '')
                              + f'  {v["holdout_mIoU_1024"]:.4f}')
    chosen = max(cands, key=lambda n: (round(cands[n]['holdout_mIoU_1024'], 6), n == 'best'))
    log(f'   chosen: {chosen}')
    if chosen == 'best': sd = {k: v.float() for k, v in best['model'].items()}
    else:
        sd = average_epochs(rdir, cands[chosen]['epochs'])
        wpath = os.path.join(rdir, f'weights_{chosen}_v1_8.pt')
        if not os.path.exists(wpath):
            save_ckpt({'model': {k: v.half() for k, v in sd.items()}, 'epochs': cands[chosen]['epochs'],
                       'holdout_mIoU_1024': cands[chosen]['holdout_mIoU_1024']}, wpath)
    model.load_state_dict(sd); model.eval(); del sd, best
    pred = Predictor(cfg, model, dev, tta=True)

    # 2. per-class thresholds on the hold-out part (with TTA), with a 2-fold check inside the hold-out part
    log('2. hold-out part with TTA (scales ' + ', '.join(f'{s:g}' for s in RULES['tta_scales']) + ' and flip) ...')
    h = threshold_hist(pred, ho, dev, os.path.join(fdir, f'holdout_tta_{chosen}'))
    NB = RULES['final_threshold_bins']; k50 = np.full(C, NB // 2); hs = h.sum(0)
    ho_fixed = float(np.nanmean(hist_iou(hs, k50))); k_all = tune_thresholds(hs); ho_tuned = float(np.nanmean(hist_iou(hs, k_all)))
    fold = np.random.default_rng(RULES['holdout_seed']).permutation(len(h)) % RULES['final_threshold_folds']; gains = []
    for f in range(RULES['final_threshold_folds']):
        k_f = tune_thresholds(h[fold != f].sum(0)); hf = h[fold == f].sum(0)
        gains.append(float(np.nanmean(hist_iou(hf, k_f)) - np.nanmean(hist_iou(hf, k50))))
    use = float(np.mean(gains)) > 0
    thr = {CLASSES[c]: k_all[c] / NB for c in range(C)}
    log(f'   hold-out mIoU at long side 1024 with TTA: {ho_fixed:.4f} at 0.5; {ho_tuned:.4f} with per-class thresholds '
        f'(chosen on the same images, optimistic)')
    log(f'   2-fold check (choose on one half, score on the other): gains ' + ', '.join(f'{g:+.4f}' for g in gains)
        + f', mean {np.mean(gains):+.4f} -> per-class thresholds ' + ('reported as a secondary result' if use else 'not reported'))
    if use: log('   thresholds: ' + ', '.join(f'{c} {t:.2f}' for c, t in thr.items() if t != 0.5) + '; all others 0.5')

    # 3. validation split, once
    sets = {'fixed': None}
    if use: sets['tuned'] = [float(np.log(t / (1 - t))) for t in thr.values()]
    log('3. validation split with TTA ' + ('(threshold 0.5 and per-class thresholds)' if use else '(threshold 0.5)') + ' ...')
    res, recs, names = evaluate_final(pred, val, dev, os.path.join(fdir, f'validation_tta_{chosen}'), sets)
    for k in sets:
        save_json(res[k], os.path.join(rdir, f'results_validation_v1_8_final_{k}.json'))
        np.savez(os.path.join(rdir, f'per_image_validation_v1_8_final_{k}.npz'), names=np.array(names),
                 **{q: np.stack([r[q] for r in recs[k]]) for q in recs[k][0]})
    head = 'fixed'                                                     # S1 section 8: main result at 0.5
    out = {'run': a.run_name, 'code': VERSION, 'rules': RULES['version'], 'weights': {'candidates': cands, 'chosen': chosen},
           'tta_scales': RULES['tta_scales'],
           'thresholds': {'holdout_mIoU_1024_tta_fixed': round(ho_fixed, 4), 'holdout_mIoU_1024_tta_tuned_in_sample': round(ho_tuned, 4),
                          'twofold_gains': [round(g, 4) for g in gains], 'used': use, 'per_class': thr},
           'validation': res, 'headline': head}
    save_json(out, os.path.join(rdir, 'final_v1_8.json'))
    log(f'\nFINAL {a.run_name} (weights {chosen}, TTA), validation split:')
    for k in sets:
        r = res[k]
        log(f'  {"per-class thresholds" if k == "tuned" else "threshold 0.5":20s} mIoU native {r["iu_native"]["mIoU"]:.4f} '
            f'(T {r["iu_native"]["T_mIoU"]:.4f}) | 512x512 {r["iu512"]["mIoU"]:.4f} (T {r["iu512"]["T_mIoU"]:.4f}) | '
            f'toolkit protocol {r["iu512_toolkit"]["mIoU"]:.4f} | clDice crack {r["clDice"]["Crack"]:.4f}')
    log('  main result (S1): threshold 0.5' + ('; per-class thresholds are a secondary result' if use else ''))
    log('per class (validation): native | 512x512 | toolkit protocol' + ('   (threshold 0.5 / per-class thresholds)' if use else ''))
    for c in CLASSES:
        cell = lambda q: ' / '.join(f'{res[k][q]["per_class"][c]}' for k in sets)
        log(f'  {c:15s} {cell("iu_native")} | {cell("iu512")} | {cell("iu512_toolkit")}')
    log(f'written final_v1_8.json, results_validation_v1_8_final_*.json, per_image_validation_v1_8_final_*.npz, '
        f'final_v1_8/ (predictions of every validation image)' + (f', weights_{chosen}_v1_8.pt' if chosen != 'best' else '')
        + f' in {rdir}')


def fit_overlap_prior(a, fit):
    """v1.9: directed pixel overlap share P[a, b] = |a and b| / |a| of the classes in the labels of the fit part (hold-out
    and validation images not used), labels drawn at long side 1024 as in s0b; diagonal 1, classes without pixels 0.
    Computed once (a few minutes on the CPU) and cached in <root>/paperc/lrd_prior_fit.json."""
    path = os.path.join(a.out, 'lrd_prior_fit.json')
    names = sorted(m['name'] for m in fit)
    if os.path.exists(path):
        js = json.load(open(path))
        if js.get('n_images') == len(names) and js.get('first') == names[0] and js.get('last') == names[-1]:
            return np.asarray(js['P'])
    L = RULES['global_long_side']; inter = np.zeros((C, C)); area = np.zeros(C); t0 = time.time()
    for n, m in enumerate(fit):
        ps = polygons(json.load(open(m['json'])))
        if not ps: continue
        s = L / max(m['W'], m['H']); w, h = max(1, round(m['W'] * s)), max(1, round(m['H'] * s))
        flat = rasterize_window(ps, s, 0, 0, w, h).reshape(C, -1); ar = flat.sum(1); area += ar
        idx = np.flatnonzero(ar)
        for i in idx: inter[i, idx] += (flat[i] & flat[idx]).sum(1)
        if (n + 1) % 1000 == 0: log(f'  LRD prior: {n + 1} / {len(fit)} fit images ({time.time() - t0:.0f} s)')
    with np.errstate(invalid='ignore', divide='ignore'): P = np.nan_to_num(inter / area[:, None])
    np.fill_diagonal(P, 1.0)
    save_json({'code': VERSION, 'n_images': len(names), 'first': names[0], 'last': names[-1], 'classes': CLASSES,
               'P': P.round(5).tolist()}, path)
    return P


def step_run(a, root):
    """train (resumable), select the best epoch on the hold-out part, then evaluate the hold-out part and the validation
    split with every S1 metric, save all predictions and an example figure. Rerunning the same command continues
    where it stopped and never repeats finished work."""
    ensure_smp()
    torch, nn, F = torch_mods()
    cfg = dict(CONFIGS[a.model]); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cont = containment_used(a) and bool(cfg.get('ovl')) and not cfg.get('no_cont')   # v1.2 / v1.10 (B3-ovlonly-L)
    if cfg.get('ovl') and a.ovl_weight is None: sys.exit('give --ovl-weight (S1 grid 0.1, 0.3, 1.0)')
    ovl_w = float(a.ovl_weight or 0.0) if (cfg.get('ovl') or cfg.get('stage') == 2) else 0.0
    if cfg.get('stage') == 2:
        name = a.name or f'Full-on-{a.stage1}-skel{a.skel}' + (f'-ovl{ovl_w:g}' if ovl_w else '') + f'-s{a.seed}'
    else:
        name = a.name or f'{a.model}{ovl_w:g}-s{a.seed}' if ovl_w else a.name or f'{a.model}-s{a.seed}'   # B3-ovl0.3-s0, B3-s0
    if a.screen: name += '-screen'                                     # screening run: hold-out only, no final evaluation
    rdir, ldir = os.path.join(a.out, 'runs', name), os.path.join(a.work, 'runs', name)
    old = os.path.join(rdir, 'config.json')                            # v1.4/v1.6: runs of earlier versions are kept aside
    if os.path.exists(old):
        oc = json.load(open(old)); code = (oc.get('code_history') or [oc.get('code')])[-1]
        if ovl_w and code in OVERLAP_FP16_BUG: move_aside((rdir, ldir), f'_invalid_{code[:-3]}')
        elif code in OLD_DICE: move_aside((rdir, ldir), f'_olddice_{code[:-3]}')
    ck = None                                                          # v1.4: never continue from NaN weights
    for src in (os.path.join(ldir, 'last.pt'), os.path.join(rdir, 'last.pt')):
        if os.path.exists(src) and not os.path.exists(os.path.join(rdir, 'done.json')):
            ck = torch.load(src, map_location=dev, weights_only=False)
            if finite_state(ck['model']): ck['src'] = src
            else: print(f'{src} contains NaN weights', flush=True); move_aside((rdir, ldir), '_invalid_nan'); ck = None
            break
    os.makedirs(rdir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    global LOG_PATH; LOG_PATH = os.path.join(rdir, 'log.txt'); log(f'--- {VERSION} | {time.strftime("%Y-%m-%d %H:%M")}')
    if os.path.exists(os.path.join(rdir, 'done.json')):
        log(f'{name}: already finished; results in {rdir}'); log(json.dumps(json.load(open(os.path.join(rdir, 'done.json'))), indent=1)); return
    fit, ho, val = (relink(x, root) for x in load_index(a))
    if a.max_images: fit, ho, val = fit[:a.max_images], ho[:max(2, a.max_images // 4)], val[:max(2, a.max_images // 4)]
    epochs = a.epochs or (RULES['epochs_B1'] if a.model == 'B1' else None)
    if epochs is None: sys.exit('give --epochs (fixed after the B1 timing run, S1 section 7)')
    batch, accum = a.batch or cfg['batch'], a.accum or cfg['accum']
    meta = {'run': name, 'model': a.model, 'config': cfg, 'epochs': epochs, 'batch': batch, 'accum': accum, 'seed': a.seed,
            'overlap_weight': ovl_w, 'containment': cont, 'skeleton': a.skel if cfg.get('stage') == 2 else None,
            'stage1': a.stage1, 'rules': RULES['version'], 'code': VERSION, 'device': str(dev),
            'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', 'n_fit': len(fit), 'n_holdout': len(ho),
            'n_validation': len(val)}
    if os.path.exists(os.path.join(rdir, 'config.json')):          # v1.5: keep the code versions of a continued run
        prev = json.load(open(os.path.join(rdir, 'config.json')))
        hist = prev.get('code_history', [prev.get('code')])
        meta['code_history'] = hist + ([VERSION] if hist[-1] != VERSION else [])
    save_json(meta, os.path.join(rdir, 'config.json')); log(json.dumps(meta, indent=1))
    torch.manual_seed(a.seed)
    gmodel = gcfg = None
    if cfg.get('stage') == 2:
        if not a.stage1: sys.exit('give --stage1 <run name of the global path, e.g. B3-ovl0.3-s0>')
        gmodel, gcfg = load_stage1(a, a.stage1, dev)
        model = make_refine_head().to(dev); data = make_tile_set(fit, a.seed, a.skel)
        groups = [{'params': model.parameters(), 'lr': RULES['optimizer']['lr_head']}]
    else:
        model = build_model(cfg, pretrained=not a.small, small=a.small).to(dev); data = make_train_set(fit, cfg, a.seed)
        enc = list(model.encoder.parameters()); ids = {id(p) for p in enc}
        lrd = [p for n_, p in model.named_parameters() if n_.startswith('lrd.')]; lrd_ids = {id(p) for p in lrd}
        groups = [{'params': enc, 'lr': cfg.get('lr_encoder', RULES['optimizer']['lr_encoder'])},
                  {'params': [p for p in model.parameters() if id(p) not in ids and id(p) not in lrd_ids],
                   'lr': cfg.get('lr_head', RULES['optimizer']['lr_head'])}]
        if lrd: groups.append({'params': lrd, 'lr': cfg.get('lr_lrd', cfg.get('lr_head', RULES['optimizer']['lr_head']))})
        if hasattr(model, 'prior'):                                    # v1.9: overlap prior of the LRD from the fit part
            P = fit_overlap_prior(a, fit) if cfg.get('lrd_prior') else np.eye(C)
            model.prior.copy_(torch.as_tensor(P, dtype=torch.float32, device=dev))
            if cfg.get('lrd_prior'):
                top = sorted(((i, j) for i in range(C) for j in range(C) if i != j), key=lambda t: -P[t])[:6]
                log('LRD prior (directed overlap on the fit part), largest: ' + ', '.join(f'{CLASSES[i]}->{CLASSES[j]} {P[i, j]:.2f}' for i, j in top))
            else: log('LRD prior off (identity): ablation without the measured overlap')
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(groups, weight_decay=cfg.get('weight_decay', RULES['optimizer']['weight_decay']))
    for g in opt.param_groups: g['base_lr'] = g['lr']
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda' and cfg.get('amp') != 'bf16')
    if a.grad_ckpt and hasattr(model, 'encoder') and hasattr(model.encoder, 'set_grad_checkpointing'):
        model.encoder.set_grad_checkpointing(True); log('gradient checkpointing on (less memory, about 30% slower)')
    steps_per_epoch = len(fit) // (batch * accum); total = max(1, steps_per_epoch * epochs); warm = max(1, steps_per_epoch)
    if a.max_iters: steps_per_epoch = min(steps_per_epoch, a.max_iters); total = steps_per_epoch * epochs
    need = nparam * 2 * epochs + nparam * 12 + (2e9 if not a.max_images else 0)
    try:
        free = shutil.disk_usage(a.out).free; log(f'Drive: {free / 1e9:.1f} GB free; this run needs about {need / 1e9:.1f} GB '
                                                  f'(fp16 weights of every epoch, last full state, predictions)')
        if free < need: log('WARNING: not enough Drive space for all checkpoints and predictions; free space before the end of the run')
    except OSError: pass
    state = {'epoch': 0, 'step': 0, 'best': -1.0, 'best_epoch': -1, 'history': []}
    lpath, dpath = os.path.join(ldir, 'last.pt'), os.path.join(rdir, 'last.pt')
    if ck:
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler']); state = ck['state']
        log(f'resumed from {ck["src"]}: epoch {state["epoch"]}, step {state["step"]} of this epoch'); del ck

    def stop_if_nan(where):
        """v1.4: stop at once (nothing saved) if the weights became NaN; the last good checkpoint stays as it was."""
        if not finite_state(model.state_dict()):
            log(f'STOPPED: the weights became NaN ({where}); no checkpoint was written after this point. '
                f'Non-finite losses so far in this epoch: {nonfinite}. Please paste this log.')
            sys.exit(1)

    def lr_at(it):
        if it < warm: return (it + 1) / warm
        return (1 - (it - warm) / max(1, total - warm)) ** RULES['optimizer']['poly_power']

    pred = Predictor(cfg, model, dev, gmodel, gcfg)
    pat = RULES['screening_patience']
    while state['epoch'] < epochs:
        if a.screen and state['best_epoch'] > 0 and state['epoch'] - state['best_epoch'] >= pat:
            log(f'early stop (screening): no hold-out improvement in the last {pat} epochs; best epoch {state["best_epoch"]}')
            break
        ep = state['epoch']; data.epoch = ep
        g = torch.Generator(); g.manual_seed(a.seed * 1000 + ep)
        order = torch.randperm(len(data), generator=g).tolist()[state['step'] * batch * accum:]
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, order), batch_size=batch, shuffle=False,
                                             num_workers=a.workers, pin_memory=dev.type == 'cuda', drop_last=True,
                                             persistent_workers=False)
        model.train(); opt.zero_grad(set_to_none=True); t0 = time.time(); tlast = t0; run_loss, nl, nb = 0.0, 0, 0
        nonfinite, n_ok = 0, 0
        for bi, b in enumerate(loader):
            if state['step'] >= steps_per_epoch: break
            b = {k: (v.to(dev, non_blocking=True) if hasattr(v, 'to') else v) for k, v in b.items()}
            with torch.autocast(dev.type, dtype=amp_dtype(cfg), enabled=dev.type == 'cuda'):
                if cfg.get('stage') == 2:
                    from torchvision.ops import roi_align
                    with torch.no_grad(), torch.autocast(dev.type, dtype=amp_dtype(gcfg), enabled=dev.type == 'cuda'):
                        gl = gmodel(b['gimg'])
                        gl = (gl[0] if isinstance(gl, tuple) else gl).float()
                    T = RULES['refine_tile']
                    boxes = torch.cat([torch.arange(len(gl), device=dev)[:, None].float(), b['box']], 1)
                    ctx = roi_align(gl, boxes, (T // 2, T // 2), 1.0, 2, True)
                    out = model(b['tile'], ctx)
                else:
                    out = model(b['img'])
            # v1.4: losses in 32-bit, outside mixed precision
            if cfg.get('stage') == 2: loss, parts = tile_loss(out.float(), b, a.skel, ovl_w)
            else: loss, parts = global_loss(cfg, out, b, ovl_w, cont)
            if torch.isfinite(loss):
                scaler.scale(loss / accum).backward(); n_ok += 1; run_loss += float(loss.detach()); nl += 1
            else:                                                      # v1.4: a non-finite loss is skipped and counted
                nonfinite += 1
                if nonfinite <= 3: log(f'  WARNING: non-finite loss at epoch {ep + 1} step {state["step"]}: {parts}; batch skipped')
                if nonfinite > max(20, steps_per_epoch * accum // 50):
                    log(f'STOPPED: {nonfinite} non-finite losses in epoch {ep + 1}; no checkpoint was written after '
                        'this point. Please paste this log.'); sys.exit(1)
            if (bi + 1) % accum == 0:
                it = ep * steps_per_epoch + state['step']
                for gr in opt.param_groups: gr['lr'] = gr['base_lr'] * lr_at(it)
                if n_ok:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), RULES['optimizer']['grad_clip'])
                    scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); state['step'] += 1; nb += 1; n_ok = 0
                if state['step'] in (5, 20) or state['step'] % 100 == 0:
                    stop_if_nan(f'epoch {ep + 1} step {state["step"]}')
                    el = time.time() - t0
                    log(f'  epoch {ep + 1}/{epochs} step {state["step"]}/{steps_per_epoch} loss {run_loss / max(1, nl):.4f} '
                        f'{parts} | {el / max(1, nb):.2f} s/step, epoch about {el / max(1, nb) * steps_per_epoch / 60:.1f} min'
                        + (f' | skipped batches {nonfinite}' if nonfinite else ''))
                if time.time() - tlast > a.save_minutes * 60:                # mid-epoch checkpoint for resuming
                    stop_if_nan(f'epoch {ep + 1} step {state["step"]}')
                    save_ckpt({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                               'state': state}, lpath, dpath); tlast = time.time()
        stop_if_nan(f'end of epoch {ep + 1}')
        # end of epoch: hold-out mIoU (long side 1024) for model selection; checkpoints of every epoch are kept
        model.eval(); miou = tmiou = None
        if (ep + 1) % a.eval_every == 0 or ep + 1 == epochs:
            res, _, _ = evaluate(pred, ho, dev, ldir, f'holdout_epoch{ep + 1:03d}', full=False, save_preds=False, chunk=10 ** 6)
            miou, tmiou = res['iu1024']['mIoU'], res['iu1024']['T_mIoU']
        dur = time.time() - t0
        state['history'].append({'epoch': ep + 1, 'loss': run_loss / max(1, nl), 'holdout_mIoU_1024': miou,
                                 'holdout_T_mIoU_1024': tmiou, 'minutes': round(dur / 60, 2), 'skipped_batches': nonfinite})
        log(f'epoch {ep + 1}: loss {run_loss / max(1, nl):.4f}, hold-out mIoU '
            + (f'{miou:.4f} (T {tmiou:.4f})' if miou is not None else 'not evaluated this epoch') + f', {dur / 60:.1f} min')
        half = {k: v.half() for k, v in model.state_dict().items()}
        save_ckpt({'model': half, 'epoch': ep + 1, 'holdout_mIoU_1024': miou}, os.path.join(ldir, f'epoch_{ep + 1:03d}.pt'),
                  os.path.join(rdir, f'epoch_{ep + 1:03d}.pt'))
        if miou is not None and miou > state['best']:
            state['best'], state['best_epoch'] = miou, ep + 1
            save_ckpt({'model': half, 'epoch': ep + 1, 'holdout_mIoU_1024': miou}, os.path.join(ldir, 'best.pt'),
                      os.path.join(rdir, 'best.pt'))
        state['epoch'] += 1; state['step'] = 0
        save_ckpt({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'state': state},
                  lpath, dpath)
        save_json(state['history'], os.path.join(rdir, 'history.json'))
    if a.screen:                       # screening (S1): only the hold-out choice is needed; the validation split is not touched
        res = {'run': name, 'screening': True, 'best_epoch': state['best_epoch'], 'epochs_run': state['epoch'],
               'early_stopped': state['epoch'] < epochs, 'best_holdout_mIoU_1024': state['best'],
               'history': state['history']}
        save_json(res, os.path.join(rdir, 'done.json'))
        log(f'{name} (screening) finished: best epoch {state["best_epoch"]}, hold-out mIoU {state["best"]:.4f}. '
            'No validation evaluation (by design).'); return
    # final evaluation with the best epoch
    best = torch.load(os.path.join(rdir, 'best.pt') if os.path.exists(os.path.join(rdir, 'best.pt'))
                      else os.path.join(ldir, 'best.pt'), map_location=dev)
    model.load_state_dict({k: v.float() for k, v in best['model'].items()}); model.eval()
    log(f'best epoch {best["epoch"]} (hold-out mIoU {best["holdout_mIoU_1024"]:.4f}); final evaluation ...')
    if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    results = {'run': name, 'best_epoch': best['epoch'], 'parameters_M': round(nparam / 1e6, 2),
               'flops': count_gflops(model, cfg, dev), 'history': state['history'],
               'training_hours': round(sum(h['minutes'] for h in state['history']) / 60, 2)}
    for tag, metas in (('holdout', ho), ('validation', val)):
        res, recs, names = evaluate(pred, metas, dev, rdir, tag, full=True, save_preds=True)
        results[tag] = res; save_json(res, os.path.join(rdir, f'results_{tag}.json'))
        np.savez(os.path.join(rdir, f'per_image_{tag}.npz'), names=np.array(names), **{
            k: np.stack([r[k] for r in recs]) for k in recs[0]})
        log(f'{tag}: mIoU native {res["iu_native"]["mIoU"]:.4f}, T {res["iu_native"]["T_mIoU"]:.4f}, '
            f'512x512 {res["iu512"]["mIoU"]:.4f}, clDice crack {res["clDice"]["Crack"]:.4f}'
            + (f', toolkit protocol {res["iu512_toolkit"]["mIoU"]:.4f}' if 'iu512_toolkit' in res else ''))
        if tag == 'validation':                                        # v1.6: per-class table of the validation split
            log('per class (validation): native | 512x512' + (' | toolkit protocol' if 'iu512_toolkit' in res else ''))
            for c in CLASSES:
                log(f'  {c:15s} {res["iu_native"]["per_class"][c]} | {res["iu512"]["per_class"][c]}'
                    + (f' | {res["iu512_toolkit"]["per_class"][c]}' if 'iu512_toolkit' in res else ''))
    if dev.type == 'cuda': results['peak_GPU_memory_GB'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    example_figure_run(pred, val, dev, os.path.join(rdir, 'fig_examples_validation.png'))
    save_json(results, os.path.join(rdir, 'done.json'))
    log(f'\n{name} finished. Everything is in {rdir}:')
    log('  config.json, history.json, epoch_*.pt (every epoch), best.pt, last.pt, results_holdout.json, results_validation.json,')
    log('  per_image_*.npz (per-image counts for bootstrap), eval_*/preds_*.zip (masks and probabilities of every image),')
    log('  fig_examples_validation.png, done.json')


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['s0b', 's2', 's1', 'run', 'eval', 'speed', 'final'])
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--mydrive', default='/content/drive/MyDrive')
    ap.add_argument('--work', default='/content/dacl10k')
    ap.add_argument('--data', default=None, help='unpacked dataset folder (skips the zip)')
    ap.add_argument('--zip', default=None, help='use this zip instead of searching Drive')
    ap.add_argument('--max-images', type=int, default=0, help='0 = all; >0 for a quick trial only')
    ap.add_argument('--width-sample', type=int, default=150)
    ap.add_argument('--bench', type=int, default=40)
    ap.add_argument('--model', choices=list(CONFIGS), help='run: model to train and evaluate')
    ap.add_argument('--epochs', type=int, default=0, help='run: epochs (B1: 30 by S1; others after the B1 timing)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ovl-weight', type=float, default=None, help='run: overlap-loss weight (B3-ovl: 0.1, 0.3 or 1.0)')
    ap.add_argument('--skel', default='T', choices=['none', 'T', 'crack', 'cldice'], help='run Full: thin-damage loss')
    ap.add_argument('--stage1', default=None, help='run Full: run name of the trained global path')
    ap.add_argument('--name', default=None, help='run: own run name (default from the settings)')
    ap.add_argument('--run', dest='run_name', default=None, help='eval, final: name of the finished run, e.g. B1-s0')
    ap.add_argument('--batch', type=int, default=0); ap.add_argument('--accum', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--save-minutes', type=float, default=20, help='run: mid-epoch checkpoint interval')
    ap.add_argument('--eval-every', type=int, default=1, help='run: hold-out evaluation every N epochs (and the last)')
    ap.add_argument('--tta', action='store_true', help='eval: test-time augmentation (3 scales x 2 flips), reported separately')
    ap.add_argument('--grad-ckpt', action='store_true', help='run: gradient checkpointing for the large ViT if memory is short')
    ap.add_argument('--screen', action='store_true', help='run: screening run (hold-out choice only, no final evaluation)')
    ap.add_argument('--small', action='store_true', help='test only: tiny encoders without pretrained weights')
    ap.add_argument('--max-iters', type=int, default=0, help='test only: steps per epoch')
    a = ap.parse_args()
    a.out = os.path.join(a.root, 'paperc'); os.makedirs(a.out, exist_ok=True)
    save_json(RULES, os.path.join(a.out, 's1_rules.json'))
    log(VERSION, '|', RULES['version'], '| output folder', a.out)
    if a.step == 's1': log(json.dumps(RULES, indent=1)); return
    root = setup_data(a)
    {'s0b': step_s0b, 's2': step_s2, 'run': step_run, 'eval': step_eval, 'speed': step_speed,
     'final': step_final}[a.step](a, root)


if __name__ == '__main__':
    main()
