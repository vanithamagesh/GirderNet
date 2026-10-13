"""Intersection-supervision loss (ISL), the overlap term of the paper, as a standalone PyTorch function.

For every image of a batch and every pair of classes (a, c) whose labels overlap in that image, the soft
intersection p_a * p_c of the predicted probabilities is compared with the labelled intersection y_a * y_c by a
smoothed Dice term

    L_pair(a, c) = 1 - (2 * sum_u p_a p_c y_a y_c + 1) / (sum_u p_a p_c + sum_u y_a y_c + 1),

and the loss is the mean of L_pair over all such pairs of the batch (0 if there is none). Pairs that do not overlap
in an image contribute nothing. The model of the paper is trained with L = L_per-class + 0.1 * ISL; the term adds no
parameters and does not change prediction. Identical to overlap_loss() in pipeline.py.

Example:
    prob = torch.sigmoid(logits)                       # (B, K, H, W), one sigmoid output per class
    loss = per_class_loss + 0.1 * intersection_supervision_loss(prob, target, valid)
"""
import torch


def intersection_supervision_loss(prob, target, valid=None):
    """prob: (B, K, H, W) probabilities; target: (B, K, H, W) binary labels, several classes per pixel allowed;
    valid: optional (B, 1, H, W) mask of the pixels to use (for example 0 on padding)."""
    with torch.autocast(prob.device.type, enabled=False):          # 32-bit sums: large crops overflow 16-bit floats
        B, K = prob.shape[:2]
        v = torch.ones_like(prob[:, :1]) if valid is None else valid
        pv = (prob.float() * v.float()).reshape(B, K, -1)
        yv = (target.float() * v.float()).reshape(B, K, -1)
        present = yv.amax(2) > 0                                   # classes labelled in each image
        pairs = present[:, :, None] & present[:, None, :] & torch.triu(
            torch.ones(K, K, dtype=torch.bool, device=prob.device), 1)
        bi, ai, ci = pairs.nonzero(as_tuple=True)
        if len(bi) == 0:
            return prob.sum() * 0
        yy = (yv[bi, ai] * yv[bi, ci]).sum(1)
        keep = yy > 0                                              # pairs whose labels overlap
        if not bool(keep.any()):
            return prob.sum() * 0
        bi, ai, ci, yy = bi[keep], ai[keep], ci[keep], yy[keep]
        pa, pc = pv[bi, ai], pv[bi, ci]
        pp = (pa * pc).sum(1)
        py = (pa * yv[bi, ai] * pc * yv[bi, ci]).sum(1)
        return (1 - (2 * py + 1) / (pp + yy + 1)).mean()
