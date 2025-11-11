import torch
import numpy as np
import os

def heatmap_to_coords(heatmaps):
    """
    Convert heatmap [B, num_joints, H, W] to coordinates (x, y, visible)
    Returns: list of [num_joints, 3]
    """
    B, J, H, W = heatmaps.shape
    coords = []
    for b in range(B):
        joints = []
        for j in range(J):
            hm = heatmaps[b, j]
            y, x = torch.nonzero(hm == hm.max(), as_tuple=True)
            if len(x) == 0:
                joints.append([0, 0, 0])
            else:
                joints.append([float(x[0].item()), float(y[0].item()), 1])
        coords.append(np.array(joints))
    return np.stack(coords, axis=0)  # [B, J, 3]


# ---------- Metrics ----------
def compute_oks(preds, gts, sigmas=None, in_vis_thre=0.2):
    """
    Object Keypoint Similarity (COCO metric)
    preds, gts: [B, J, 3] (x, y, vis)
    """
    if sigmas is None:
        sigmas = np.ones(preds.shape[1]) * 0.05
    vars = (sigmas * 2) ** 2
    oks_all = []
    for pred, gt in zip(preds, gts):
        vis = gt[:, 2] > in_vis_thre
        if not np.any(vis):
            oks_all.append(0)
            continue
        d = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
        oks = np.mean(np.exp(-d[vis] ** 2 / (2 * vars[vis] * (np.max(d) + np.finfo(float).eps) ** 2)))
        oks_all.append(oks)
    return np.mean(oks_all)


def compute_pck(preds, gts, threshold=0.05):
    """Percentage of Correct Keypoints"""
    correct = 0
    total = 0
    for pred, gt in zip(preds, gts):
        head_size = np.linalg.norm(gt[0, :2] - gt[1, :2]) + 1e-6
        d = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
        correct += np.sum(d / head_size < threshold)
        total += len(gt)
    return correct / total


def compute_pcp(preds, gts, limbs):
    """Percentage of Correct Parts"""
    correct = 0
    total = 0
    for pred, gt in zip(preds, gts):
        for (i, j) in limbs:
            pd = np.linalg.norm(pred[i, :2] - pred[j, :2])
            gd = np.linalg.norm(gt[i, :2] - gt[j, :2])
            err = abs(pd - gd) / (gd + 1e-6)
            correct += (err < 0.5)
            total += 1
    return correct / total


def compute_pdj(preds, gts, threshold=0.05):
    """Percentage of Detected Joints"""
    correct = 0
    total = 0
    for pred, gt in zip(preds, gts):
        torso = np.linalg.norm(gt[5, :2] - gt[12, :2]) + 1e-6  # example: left shoulder–right hip
        d = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
        correct += np.sum(d / torso < threshold)
        total += len(gt)
    return correct / total


def compute_all_metrics(preds, gts):
    limbs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]  # example skeleton
    oks = compute_oks(preds, gts)
    pck = compute_pck(preds, gts)
    pcp = compute_pcp(preds, gts, limbs)
    pdj = compute_pdj(preds, gts)
    return {"OKS": oks, "PCK": pck, "PCP": pcp, "PDJ": pdj}
