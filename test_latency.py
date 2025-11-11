from models.pose_higher_HRNet import PoseHigherResolutionNet
import torch
import time
import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


if __name__ == "__main__":
    import types
    from types import SimpleNamespace
    try:
        with open("w32_512_adam_lr1e-3.yaml", "r") as f:
            cfg_dict = yaml.safe_load(f)

        # Convert to namespace
        cfg = dict_to_namespace(cfg_dict)

        # Now you can safely use dot access
        print(cfg.MODEL)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")

    # ---- build model ----
    model = PoseHigherResolutionNet(cfg)
    model.eval()

    # ---- create dummy input ----
    dummy = torch.ones((1, 3, 256, 192)) * 0.2  # [B, C, H, W]
    start = time.time()
    outputs = model(dummy)
    end = time.time()

    # ---- print summary ----
    print("\nModel Summary:")
    print("=" * 60)
    print(f"Input shape: {tuple(dummy.shape)}")
    print(f"Number of output stages: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f" → Output[{i}] shape: {tuple(out.shape)}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
    print("=" * 60)
