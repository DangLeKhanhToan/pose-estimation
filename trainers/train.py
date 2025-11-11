import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# print(sys.path)

from utils.metrics import heatmap_to_coords, compute_all_metrics
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device="cuda",
    run_id=0,
):
    os.makedirs(f"runs/train{run_id}", exist_ok=True)
    train_log = {"loss": [], "OKS": [], "PCK": [], "PCP": [], "PDJ": []}
    val_log = {"loss": [], "OKS": [], "PCK": [], "PCP": [], "PDJ": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds, all_gts = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for imgs, gt_keypoints in pbar:
            imgs = imgs.to(device)
            gt_keypoints = gt_keypoints.numpy()  # [B, 33, 3]

            preds = model(imgs)
            loss = criterion(preds, torch.zeros_like(preds).to(device))  # placeholder, replace with GT heatmaps

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Convert heatmaps → keypoints
            preds_np = heatmap_to_coords(preds.detach().cpu())
            all_preds.append(preds_np)
            all_gts.append(gt_keypoints)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
        metrics = compute_all_metrics(all_preds, all_gts)

        train_log["loss"].append(total_loss / len(train_loader))
        for k, v in metrics.items():
            train_log[k].append(v)

        # ---------- VALIDATE every 3 epochs ----------
        if (epoch + 1) % 3 == 0:
            val_loss, val_metrics = validate_model(model, val_loader, criterion, device)
            val_log["loss"].append(val_loss)
            for k, v in val_metrics.items():
                val_log[k].append(v)

        save_graph(train_log, val_log, f"runs/train{run_id}/metrics.png")

    return train_log, val_log


def validate_model(model, loader, criterion, device="cuda"):
    model.eval()
    total_loss = 0
    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, gt_keypoints in loader:
            imgs = imgs.to(device)
            gt_keypoints = gt_keypoints.numpy()
            preds = model(imgs)
            loss = criterion(preds, torch.zeros_like(preds).to(device))
            total_loss += loss.item()
            preds_np = heatmap_to_coords(preds.detach().cpu())
            all_preds.append(preds_np)
            all_gts.append(gt_keypoints)

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    metrics = compute_all_metrics(all_preds, all_gts)
    return total_loss / len(loader), metrics


def save_graph(train_log, val_log, path):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_log["loss"], label="Train Loss")
    if len(val_log["loss"]) > 0:
        plt.plot(np.arange(2, len(val_log["loss"]) * 3 + 1, 3), val_log["loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(2, 1, 2)
    for k in ["OKS", "PCK", "PCP", "PDJ"]:
        plt.plot(train_log[k], label=f"Train {k}")
        if len(val_log[k]) > 0:
            plt.plot(np.arange(2, len(val_log[k]) * 3 + 1, 3), val_log[k], label=f"Val {k}")
    plt.legend()
    plt.title("Metrics Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



from models.pose_model import PoseModel
from dataloader.yolo_dataset import YOLOLoader
from trainers.loss import HeatmapLoss
from trainers.optimize import create_optimizer
from trainers.train import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = torch.utils.data.DataLoader(
    YOLOLoader("data/images/train", "data/labels/train"),
    batch_size=8, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    YOLOLoader("data/images/val", "data/labels/val"),
    batch_size=8, shuffle=False
)

model = PoseModel("efficientnetv2", num_joints=33).to(device)
criterion = HeatmapLoss()
optimizer = create_optimizer(model, {"optimizer": "adam", "lr": 1e-3})

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, device=device, run_id=1)
