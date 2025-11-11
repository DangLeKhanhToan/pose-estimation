import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import time

BN_MOMENTUM = 0.1


class PoseEfficientNetV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_joints = cfg['MODEL']['NUM_JOINTS']

        # --- Load EfficientNetV2 backbone ---
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.features.children()))
        self.backbone_out_channels = backbone.classifier[1].in_features  # 1280

        # --- Deconv head ---
        self.deconv1 = nn.ConvTranspose2d(
            self.backbone_out_channels, 256, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        self.deconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)

        # --- Final heatmap conv ---
        self.final_layer = nn.Conv2d(
            128, self.num_joints, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    import yaml

    try:
        with open("config/efficientV2_backbone.yaml", "r") as f:
            cfg = yaml.safe_load(f)

    except Exception as e:
       print(f"Error loading config file: {e}")

    model = PoseEfficientNetV2(cfg)
    model.eval()

    dummy = torch.ones((1, 3, 256, 192)) * 0.2

    # --- Warm-up ---
    for _ in range(5):
        _ = model(dummy)

    # --- Measure latency ---
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.time() - start) * 1000

    print("\nModel Summary:")
    print("=" * 60)
    print(f"Backbone: EfficientNetV2-S")
    print(f"Input shape: {tuple(dummy.shape)}")
    print(f"Output shape: {_ .shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Latency: {elapsed:.2f} ms")
    print("=" * 60)
