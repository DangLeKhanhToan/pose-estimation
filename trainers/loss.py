import torch
import torch.nn as nn

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, preds, targets):
        # preds: (B, num_joints, H, W)
        # targets: (B, num_joints, H, W)
        return self.criterion(preds, targets)

class JointMSELoss(nn.Module):
    """MSE Loss with joint visibility weighting"""
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
    
    def forward(self, preds, targets, target_weight=None):
        batch_size = preds.size(0)
        num_joints = preds.size(1)
        
        heatmaps_pred = preds.reshape((batch_size, num_joints, -1))
        heatmaps_gt = targets.reshape((batch_size, num_joints, -1))
        
        loss = self.criterion(heatmaps_pred, heatmaps_gt)
        
        if self.use_target_weight and target_weight is not None:
            loss = loss * target_weight.reshape((batch_size, num_joints, 1))
        
        return loss.mean()
    
class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss - Better for heatmap regression
    Paper: "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, preds, targets):
        delta = (targets - preds).abs()
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, 
                                              self.alpha - targets))) * \
            (self.alpha - targets) * torch.pow(self.theta / self.epsilon, 
                                                self.alpha - targets - 1) * \
            (1 / self.epsilon)
        
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(
            self.theta / self.epsilon, self.alpha - targets))
        
        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, 
                                                   self.alpha - targets)),
            A * delta - C
        )
        
        return losses.mean()