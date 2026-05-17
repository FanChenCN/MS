import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def adjusted_rand_index(true_ids, pred_ids, ignore_background=False):
    """
    Args:
        true_ids: (B, H, W) integer tensor
        pred_ids: (B, H, W) integer tensor
        ignore_background: if True, ignore pixels where true_ids == 0

    Returns:
        ARI score, scalar tensor
    """
    if len(true_ids.shape) == 3:
        true_ids = true_ids.unsqueeze(1)
    if len(pred_ids.shape) == 3:
        pred_ids = pred_ids.unsqueeze(1)

    true_oh = F.one_hot(true_ids).float()
    pred_oh = F.one_hot(pred_ids).float()

    if ignore_background:
        true_oh = true_oh[..., 1:]

    N = torch.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
    A = N.sum(dim=-1)
    B = N.sum(dim=-2)
    num_points = A.sum(dim=1)

    rindex = (N * (N - 1)).sum(dim=[1, 2])
    aindex = (A * (A - 1)).sum(dim=1)
    bindex = (B * (B - 1)).sum(dim=1)
    expected_rindex = aindex * bindex / torch.clamp(num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    return torch.where(denominator != 0, ari, torch.ones_like(ari)).mean()


def fARI(true_mask, pred_mask):
    """Foreground Adjusted Rand Index, ignores background (label 0).

    Args:
        true_mask: (B, H, W) integer tensor
        pred_mask: (B, H, W) integer tensor

    Returns:
        scalar float
    """
    return adjusted_rand_index(true_mask, pred_mask, ignore_background=True).item()


def ARI(true_mask, pred_mask):
    """Full Adjusted Rand Index (including background).

    Args:
        true_mask: (B, H, W) integer tensor
        pred_mask: (B, H, W) integer tensor

    Returns:
        scalar float
    """
    return adjusted_rand_index(true_mask, pred_mask, ignore_background=False).item()


def _hungarian_miou_single(gt_mask, pred_mask, ignore_background=True):
    """Single-sample Hungarian-matched IoU.

    Args:
        gt_mask: (H*W,) integer tensor, 0 = background
        pred_mask: (H*W,) integer tensor

    Returns:
        float or np.nan
    """
    if gt_mask.max().item() == 0 and ignore_background:
        return np.nan

    true_oh = F.one_hot(gt_mask).float()
    if ignore_background:
        true_oh = true_oh[..., 1:]
    pred_oh = F.one_hot(pred_mask).float()
    N, M = true_oh.shape[-1], pred_oh.shape[-1]

    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect
    iou = (intersect / (union + 1e-8)).detach().cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    if M >= N:
        return iou[row_ind, col_ind].mean()
    return iou[row_ind, col_ind].sum() / float(N)


def hungarian_miou(gt_mask, pred_mask):
    """Batch-level Hungarian-matched mIoU (foreground only).

    Args:
        gt_mask: (B, H, W) integer tensor
        pred_mask: (B, H, W) integer tensor

    Returns:
        scalar float
    """
    gt_flat = gt_mask.flatten(1, 2)
    pred_flat = pred_mask.flatten(1, 2)
    ious = []
    for i in range(gt_flat.shape[0]):
        iou = _hungarian_miou_single(gt_flat[i], pred_flat[i], ignore_background=True)
        ious.append(iou)
    return float(np.nanmean(ious))


def _mean_best_overlap_single(gt_mask, pred_mask):
    """Single-sample Mean Best Overlap (foreground only).

    Args:
        gt_mask: (H*W,) integer tensor, 0 = background
        pred_mask: (H*W,) integer tensor

    Returns:
        float or np.nan
    """
    if gt_mask.max().item() == 0:
        return np.nan

    true_oh = F.one_hot(gt_mask).float()
    true_oh = true_oh[..., 1:]
    pred_oh = F.one_hot(pred_mask).float()

    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect
    iou = (intersect / (union + 1e-8)).detach().cpu().numpy()

    return iou.max(axis=1).mean()


def mean_best_overlap(gt_mask, pred_mask):
    """Batch-level Mean Best Overlap (MBO), foreground only.

    Args:
        gt_mask: (B, H, W) integer tensor
        pred_mask: (B, H, W) integer tensor

    Returns:
        scalar float
    """
    gt_flat = gt_mask.flatten(1, 2)
    pred_flat = pred_mask.flatten(1, 2)
    mbos = []
    for i in range(gt_flat.shape[0]):
        mbo = _mean_best_overlap_single(gt_flat[i], pred_flat[i])
        mbos.append(mbo)
    return float(np.nanmean(mbos))