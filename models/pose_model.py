import torch
import torchvision
import torch.nn as nn
import time

from .backbones.EfficientNetV2 import EfficientNetV2Backbone
from .backbones.ResNet import ResNetBackbone
# from backbones.HRNet import HRNetBackbone
# from backbones.HigherHRNet import HigherHRNetBackbone


BN_MOMENTUM = 0.1

# ====== 5. MODEL (EfficientNetV2-S backbone + deconv to stride 4) ======
class HeatmapPoseNet(nn.Module):
    def __init__(self, num_kpts=33, out_stride=4):
        super().__init__()
        # --- EfficientNetV2-S backbone ---
        backbone = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        # Output của features cuối: (B, 1280, H/32, W/32)
        self.stem = backbone.features  # giữ nguyên phần trích đặc trưng

        # --- Upsampling decoder (stride 32 → 4) ---
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, num_kpts, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)    # (B,1280,H/32,W/32)
        x = self.deconv(x)  # (B,128,H/4,W/4)
        x = self.head(x)    # (B,K,H/4,W/4)
        return torch.sigmoid(x)




class PoseModel(nn.Module):
    def __init__(self, backbone_name="efficientnetv2", num_joints=33, out_stride=4):
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
        x = self.final_layer(x)
        return x


def test_latency(backbone="efficientnetv2", device="cuda" if torch.cuda.is_available() else "cpu"):
    if backbone.lower() not in ["efficientnetv2", "resnet", "heatmappose"]:
        raise ValueError(f"Backbone {backbone} not supported for latency test.")
    elif backbone.lower() == "heatmappose":
        model = HeatmapPoseNet(num_kpts=33).to(device)
    else:
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
    print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")


if __name__ == "__main__":
    for name in ["efficientnetv2", "resnet","heatmappose"]:
        test_latency(name)

