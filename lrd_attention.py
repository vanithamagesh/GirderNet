"""Paper C: segmentations and attention maps of the label-relation decoder (LRD). Version 1.0, 12 October 2026.
GPU runtime (A100: about 10 min, most of it reading the 975 validation images); the trained LRD is only run forward.

Uses the final LRD weights of run B3-lrd-L0.1-s0 (the weights chosen by the final step) and its stored final
predictions. For each of the 88 validation images of the segmentation figures (panels v1.1, v1.2 and v1.3) it writes
<stem>_lrd.npz with
  lrd         final LRD masks at long side 1024 (TTA, threshold 0.5, as stored by the final step), bits packed along
              the width, the same grid as the panel masks
  cross       (layers, 19, h, w) uint8: cross-attention of every class query to the image positions of its layer
              (layer 1 stride 32, layer 2 stride 16, layer 3 stride 8), mean over heads, each map scaled to its maximum
  cross_max   (layers, 19) float32: the maximum of each map before scaling
  rel         (layers, 19, 19) float32: class-to-class attention of every layer (row: attending class), mean over heads
  stride, shape
and, over all 975 validation images, relations_validation.npz: the mean class-to-class attention of every layer, the
overlap prior of the fit part, and the learned bias weights (alpha per head, R) of every layer.
The attention is computed on the image at long side 1024 without TTA, as one forward pass of the final weights.
Writes <root>/paperc/lrd_attention_v1_0/ and lrd_attention_v1_0.zip (about 15-25 MB). Upload the zip.
Needs pipeline.py and figs_helpers.py in the same folder (both on Drive).
Colab (GPU runtime):
  from google.colab import drive; drive.mount('/content/drive')
  !python lrd_attention.py
"""
import argparse, importlib.util, json, os, shutil, sys, time
import numpy as np

VERSION = 'lrd_attention.py'
RUN = 'B3-lrd-L0.1-s0'
PANEL_IMAGES = ('0016 0017 0018 0020 0021 0024 0031 0034 0039 0044 0045 0046 0060 0066 0067 0070 0073 0076 0086 0097 '
                '0102 0104 0105 0108 0111 0128 0139 0153 0179 0220 0229 0231 0237 0253 0255 0295 0310 0328 0362 0371 '
                '0373 0375 0381 0382 0394 0454 0457 0458 0460 0533 0542 0545 0556 0558 0573 0577 0581 0583 0589 0592 '
                '0622 0632 0634 0667 0669 0670 0672 0796 0803 0805 0808 0826 0837 0839 0853 0871 0873 0882 0883 0889 '
                '0898 0922 0928 0934 0946 0947 0951 0968').split()


def log(*a): print(*a, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def attention(model, x):
    """one forward pass of the LRD returning, per layer, the cross-attention (B, K, h, w) of every class query to its
    pyramid level and the class-to-class attention (B, K, K); follows ViTLRD.forward of pipeline.py."""
    y, outs = model.pixel_path(x); L = model.lrd; B = x.shape[0]
    q = L['qfeat'].weight[None].expand(B, -1, -1); qpos = L['qpos'].weight[None].expand(B, -1, -1)
    mems = model.memories(outs, x); cross, rel, strides = [], [], []
    for i, layer in enumerate(L['layers']):
        mem, mpos = mems[i % 3]; lev = (3, 2, 1)[i % 3]; hh, ww = outs[lev].shape[-2:]
        h = layer.n1(q)
        o, w = layer.cross(h + qpos, mem + mpos, mem, need_weights=True, average_attn_weights=True)
        rel.append(layer.relation_weights(q + o, qpos, model.prior).float())
        cross.append(w.float().reshape(B, -1, hh, ww)); strides.append(x.shape[-1] // ww)
        q = layer(q, qpos, mem, mpos, model.prior)
    return cross, rel, strides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/content/drive/MyDrive/PaperC_SHM_study1')
    ap.add_argument('--zip', default=None); ap.add_argument('--data', default=None)
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--images', default=None, help='test only: comma-separated image numbers instead of the 88 panel images')
    ap.add_argument('--max-images', type=int, default=0, help='test only')
    ap.add_argument('--small', action='store_true', help='test only: small encoder')
    a = ap.parse_args(); t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    pcf = next((p for p in (os.path.join(here, f) for f in ('pipeline.py', 'pipeline.py')) if os.path.exists(p)), None)
    if not pcf: sys.exit('pipeline.py not found next to this script')
    if not os.path.exists(os.path.join(here, 'figs_helpers.py')): sys.exit('figs_helpers.py not found next to this script')
    PC = load('paperc', pcf); FG = load('paperc_figs', os.path.join(here, 'figs_helpers.py')); FG.PC = PC
    PC.ensure_smp()
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rdir = os.path.join(a.root, 'paperc', 'runs', a.run)
    fj = os.path.join(rdir, 'final_v1_8.json')
    if not os.path.exists(fj): sys.exit(f'no final step of {a.run} found ({fj}); run the final step first')
    chosen = json.load(open(fj))['weights']['chosen']
    wpath = os.path.join(rdir, 'best.pt' if chosen == 'best' else f'weights_{chosen}_v1_8.pt')
    cfg = dict(PC.CONFIGS[json.load(open(os.path.join(rdir, 'config.json')))['model']])
    model = PC.build_model(cfg, pretrained=False, small=a.small)
    sd = torch.load(wpath, map_location='cpu')['model']
    model.load_state_dict({k: v.float() for k, v in sd.items()}); model.to(dev).eval(); del sd
    log(f'{VERSION}: {a.run}, weights {chosen} ({os.path.basename(wpath)}), {dev}')
    src = FG.pred_source(rdir, 'final')
    if src is None or src[2] != 'TTA': sys.exit(f'stored final predictions of {a.run} not found')
    out = os.path.join(a.root, 'paperc', 'lrd_attention_v1_0'); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    idx = json.load(open(os.path.join(a.root, 'paperc', 's2_index.json')))['index']
    val = sorted([m for m in idx if m['split'] == 'validation'], key=lambda m: m['name'])
    if a.max_images: val = val[:a.max_images]
    keep = set(a.images.split(',') if a.images else PANEL_IMAGES)
    imgs = FG.Images(a); L_ = PC.RULES['global_long_side']; K = PC.C
    on = dev.type == 'cuda' and torch.cuda.get_device_capability(dev)[0] >= 8
    rel_sum, n_img, written = None, 0, []
    for i, m in enumerate(val):
        W, H = m['W'], m['H']; s = L_ / max(W, H); w, h = max(1, round(W * s)), max(1, round(H * s))
        rgb = FG.load_rgb(imgs.read(m['name']), W, H, w, h)
        x, (hv, wv) = PC.norm_pad(rgb)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=on):
            cross, rel, strides = attention(model, x.to(dev))
        R = np.stack([r[0].cpu().numpy() for r in rel])                                   # (layers, K, K)
        rel_sum = R if rel_sum is None else rel_sum + R; n_img += 1
        stem = os.path.splitext(m['name'])[0]
        if stem[-4:] in keep:
            maps, mx = [], []
            for c, st in zip(cross, strides):
                cm = c[0, :, :-(-hv // st), :-(-wv // st)].cpu().numpy()                 # valid part, no padding
                peak = cm.reshape(K, -1).max(1).astype(np.float32); mx.append(peak)
                maps.append(np.round(255 * cm / np.maximum(peak, 1e-12)[:, None, None]).astype(np.uint8))
            mask = FG.load_mask(src, m['name'])
            if mask is None or mask.shape != (K, h, w): log(f'  {m["name"]}: stored LRD mask missing or of another size'); continue
            np.savez_compressed(os.path.join(out, f'{stem}_lrd.npz'), lrd=np.packbits(mask, axis=-1), shape=np.array(mask.shape),
                                **{f'cross{j + 1}': mp for j, mp in enumerate(maps)}, cross_max=np.stack(mx),
                                stride=np.array(strides), rel=R.astype(np.float32))
            written.append(m['name'])
        if (i + 1) % 100 == 0: log(f'  {i + 1} / {len(val)} images ({time.time() - t0:.0f} s), {len(written)} panel images written')
    layers = model.lrd['layers']
    np.savez(os.path.join(out, 'relations_validation.npz'), mean_rel=(rel_sum / max(n_img, 1)).astype(np.float32),
             prior=model.prior.float().cpu().numpy(), alpha=np.stack([l.alpha.detach().float().cpu().numpy() for l in layers]),
             R=np.stack([l.R.detach().float().cpu().numpy() for l in layers]), classes=np.array(PC.CLASSES), n_images=n_img)
    json.dump({'code': VERSION, 'run': a.run, 'weights': chosen, 'classes': list(PC.CLASSES), 'images': written,
               'validation_images': n_img, 'grid': f'long side {L_}', 'masks': 'final LRD predictions (TTA, threshold 0.5)',
               'attention': 'one forward pass at long side 1024 without TTA; mean over heads'},
              open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    z = shutil.make_archive(out, 'zip', out)
    log(f'\n{len(written)} panel images and the mean relations over {n_img} validation images written; '
        f'{z}: {os.path.getsize(z) / 1e6:.1f} MB ({time.time() - t0:.0f} s). Please upload this zip file.')


if __name__ == '__main__':
    main()
