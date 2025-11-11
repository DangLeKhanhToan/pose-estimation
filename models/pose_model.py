import torch
import torch.nn as nn
import time

from backbones.EfficientNetV2 import EfficientNetV2Backbone
from backbones.ResNet import ResNetBackbone
# from backbones.HRNet import HRNetBackbone
# from backbones.HigherHRNet import HigherHRNetBackbone


BN_MOMENTUM = 0.1

class PoseModel(nn.Module):
    def __init__(self, backbone_name="efficientnetv2", num_joints=33):
        super().__init__()

        if backbone_name.lower() == "efficientnetv2":
            self.backbone = EfficientNetV2Backbone()
        elif backbone_name.lower() == "resnet":
            self.backbone = ResNetBackbone()
        else:
            raise ValueError(f"Backbone {backbone_name} not supported yet.")

        in_ch = self.backbone.out_channels

        # ---- Unified Decoder ----
        self.deconv1 = nn.ConvTranspose2d(in_ch, 256, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.final_layer = nn.Conv2d(128, num_joints, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.relu(self.bn1(self.deconv1(x)))
        x = torch.relu(self.bn2(self.deconv2(x)))
        return self.final_layer(x)


def test_latency(backbone="efficientnetv2", device="cuda" if torch.cuda.is_available() else "cpu"):
    model = PoseModel(backbone_name=backbone).to(device)
    model.eval()

    dummy = torch.ones((1, 3, 256, 192), device=device) * 0.2

    # Warm-up
    for _ in range(5):
        _ = model(dummy)

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = (time.time() - start) * 1000

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n=== {backbone.upper()} Pose Model ===")
    print(f"Input: {tuple(dummy.shape)}")
    print(f"Output: {tuple(_.shape)}")
    print(f"Parameters: {total_params:,}")
    print(f"Latency: {elapsed:.2f} ms")


if __name__ == "__main__":
    for name in ["efficientnetv2", "resnet"]:
        test_latency(name)
