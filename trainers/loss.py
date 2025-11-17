import torch
import torch.nn as nn
import numpy as np

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, preds, targets, target_weight=None):
        if target_weight is not None:
            # apply mask per joint
            loss = ((preds - targets) ** 2) * target_weight
            return loss.mean()
        else:
            return self.criterion(preds, targets)

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
    

def generate_target(joints_3d, num_joints, heatmap_size=(64, 64), sigma=2, feat_stride=(4, 4)):
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_3d[:, 0, 1]  # visibility flag
    target = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    tmp_size = sigma * 3

    for i in range(num_joints):
        mu_x = int(joints_3d[i, 0, 0] / feat_stride[0] + 0.5)
        mu_y = int(joints_3d[i, 1, 0] / feat_stride[1] + 0.5)

        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if (ul[0] >= heatmap_size[1] or ul[1] >= heatmap_size[0] or br[0] < 0 or br[1] < 0):
            target_weight[i] = 0
            continue

        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
        img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[0])

        if target_weight[i] > 0.5:
            target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, np.expand_dims(target_weight, -1)
