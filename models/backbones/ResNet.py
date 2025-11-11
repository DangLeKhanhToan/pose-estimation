import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet18, ResNet18_Weights

BN_MOMENTUM = 0.1

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        # backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.out_channels = 2048
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # remove avgpool+fc
        

    def forward(self, x):
        return self.features(x)
