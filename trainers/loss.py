import torch
import torch.nn as nn

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, preds, targets):
        return self.criterion(preds, targets)

class AELoss(nn.Module):
    """Associative Embedding Loss"""
    def __init__(self, pull_weight=0.001, push_weight=0.001):
        super().__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight

    def forward(self, tags, joints):
        if tags.shape[0] == 0:
            return torch.tensor(0.0, device=tags.device)
        mean = torch.mean(tags, dim=0)
        pull_loss = torch.mean((tags - mean) ** 2)
        push_loss = torch.mean(torch.clamp(1 - torch.abs(tags - mean), min=0))
        return self.pull_weight * pull_loss + self.push_weight * push_loss
