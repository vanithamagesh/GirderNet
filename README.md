# GirderNet

**Overlap-aware multi-label segmentation of bridge surface damage.** A girder is the main load-carrying beam of a
bridge; GirderNet is a DINOv3 ViT-L/16 + UPerNet segmentation model trained with an intersection-supervision loss, so
that it keeps the combinations of damage that occur together, such as exposed reinforcement with rust and spalling.

Code for the paper **"Overlap-aware multi-label segmentation of bridge surface damage"** (manuscript in
preparation for *Structural Health Monitoring*). Authors and citation will be added on publication.

Bridge inspection images often show several damage types at the same place: exposed reinforcement inside rust stains
and spalls, cracks through efflorescence, wet spots on weathered concrete. The dacl10k dataset ([toolkit](https://github.com/phiyodr/dacl10k-toolkit))
labels these as overlapping polygons, but models are usually trained with per-class losses and scored with per-class
IoU, which never check whether the right *combination* of damage is predicted at a pixel. This repository adds an
**intersection-supervision loss (ISL)** that supervises, for every pair of classes labelled together, the predicted
intersection against the labelled one, and evaluates the **exact label-set match** M on pixels with two or more labels.

## Main results (dacl10k v2 validation split, 975 images, TTA)

| | Baseline | With ISL | Difference [95% interval] |
|---|---|---|---|
| Exact label-set match M (primary) | 37.1 | **40.4** | +3.3 [+0.6, +6.0] |
| M on pixels with three or more labels (long side 1024) | 14.4 | **17.6** | +3.2 [+1.3, +5.1] |
| Pair IoU ExposedRebars & Rust | 42.8 | **44.8** | +2.0 [+0.2, +4.2] |
| mIoU, full size | 49.7 | **50.0** | +0.4 [-0.2, +0.9] |

Both models: DINOv3 ViT-L/16 + DPT reassembly + UPerNet, same data, schedule and seed; all choices made on a hold-out
part of the training images; paired bootstrap over images (2,000 resamples). Without retraining, the model with the
ISL also scores higher on the S2DS and DeepCrack test sets (`results/`).

## The loss

`isl_loss.py` is a standalone implementation (PyTorch, about 40 lines) that can be added to any multi-label
segmentation model with one sigmoid output per class:

```python
from isl_loss import intersection_supervision_loss
loss = per_class_loss + 0.1 * intersection_supervision_loss(torch.sigmoid(logits), target, valid)
```

## Repository

| File | Purpose |
|---|---|
| `pipeline.py` | data index and split (`s2`), training (`run`), evaluation (`eval`), final step with weight averaging, TTA and thresholds chosen on the hold-out part (`final`); all model configurations (B1, baseline `B3-L`, ISL `B3-ovl-L`, label-relation decoder `B3-lrd-L`) and the fixed study rules |
| `isl_loss.py` | standalone intersection-supervision loss |
| `data_check_s0.py` | dataset checks and class statistics (step S0) |
| `bootstrap.py` | paired bootstrap of two runs (`--a`, `--b`) |
| `external_eval.py` | S2DS and DeepCrack evaluation without retraining |
| `inspection_records.py` | defect lists (inspection records) derived from the label sets |
| `labelset_run.py`, `labelset_errors.py` | decomposition of label-set errors (exact, missing, extra, mixed, none) |
| `lrd_attention.py` | attention maps of the label-relation decoder |
| `export_panels_*.py`, `figs_helpers.py` | export of images and masks for the qualitative figures |
| `figures/results_figures.py` | result figures of the paper from the exact values |
| `results/` | per-class IoU, summaries, training curves and external results |

## Setup

Python 3.10+ and a GPU with at least 40 GB (an A100 was used; a 768 x 1024 crop of ViT-L/16 in bfloat16).

```bash
pip install -r requirements.txt
```

## Data

dacl10k v2 is available from its authors (see the [dacl10k toolkit](https://github.com/phiyodr/dacl10k-toolkit); licence CC BY-NC 4.0) and is not redistributed here. Download the
dev-phase archive `dacl10k_v2_devphase.zip` and place it in `<root>/dacl10k/`. S2DS (academic use only) and
DeepCrack, used for the external evaluation, are downloaded by `external_eval.py` from their public sources or read
from `<root>/xdata/`.

## Reproducing the study

All steps write to `<root>/paperc/`; `--root` defaults to a Google Drive folder for Colab.

```bash
python pipeline.py s0b --root ROOT                     # measurements that fix the study rules
python pipeline.py s2  --root ROOT                     # data index, fit / hold-out split
python pipeline.py run --root ROOT --model B3-L --epochs 20                    # baseline
python pipeline.py run --root ROOT --model B3-ovl-L --ovl-weight 0.1 --epochs 20   # with the ISL
python pipeline.py final --root ROOT --run B3-L-s0
python pipeline.py final --root ROOT --run B3-ovl-L0.1-s0
python bootstrap.py --root ROOT --a B3-ovl-L0.1-s0 --b B3-L-s0
python external_eval.py --root ROOT                    # S2DS and DeepCrack
python inspection_records.py --root ROOT
python labelset_run.py --root ROOT
```

`run` resumes after an interruption and never repeats a finished run. Training takes about 3.8 h per model on an
A100; prediction takes 0.10 s per image (0.38 s with test-time augmentation).

## Licence

To be added by the authors. The dataset keeps its own licence (CC BY-NC 4.0).
