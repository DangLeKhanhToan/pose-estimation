import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

BN_MOMENTUM = 0.1

class EfficientNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.features.children()))
        self.out_channels = backbone.classifier[1].in_features  # usually 1280

    def forward(self, x):
        return self.features(x)
