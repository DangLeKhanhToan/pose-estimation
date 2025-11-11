import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.metrics import heatmap_to_coords, compute_all_metrics
from utils.visualization import save_train_graph, save_val_graph
from models.pose_model import PoseModel
from dataloader.yolo_dataset import YOLOLoader
from trainers.loss import HeatmapLoss
from trainers.optimize import create_optimizer


# ------------------------ TRAIN LOOP ------------------------
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
            gt_keypoints = gt_keypoints.numpy()

            preds = model(imgs)
            loss = criterion(preds, torch.zeros_like(preds).to(device))  # placeholder

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds_np = heatmap_to_coords(preds.detach().cpu())
            all_preds.append(preds_np)
            all_gts.append(gt_keypoints)
            pbar.set_postfix(loss=f"{loss.item():.6f}")

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

        print("Train Log:", {k: v[-1] for k, v in train_log.items()})
        if len(val_log["loss"]) > 0:
            print("Val Log:", {k: v[-1] for k, v in val_log.items()})

        save_train_graph(train_log, f"runs/train{run_id}/train_metrics.png")
        save_val_graph(val_log, f"runs/train{run_id}/val_metrics.png")

    return train_log, val_log


# ------------------------ VALIDATION ------------------------
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


# ------------------------ ARGPARSE ENTRY ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Estimation Model")

    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training data folder (contains 'images' and 'labels')")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to validation data folder (contains 'images' and 'labels')")
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640],
                        help="Input image size (width height)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")

    return parser.parse_args()


# ------------------------ MAIN SCRIPT ------------------------
if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_img_dir = os.path.join(args.train_path, "images")
    train_label_dir = os.path.join(args.train_path, "labels")
    val_img_dir = os.path.join(args.val_path, "images")
    val_label_dir = os.path.join(args.val_path, "labels")

    train_loader = torch.utils.data.DataLoader(
        YOLOLoader(train_img_dir, train_label_dir, input_size=tuple(args.input_size)),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        YOLOLoader(val_img_dir, val_label_dir, input_size=tuple(args.input_size)),
        batch_size=args.batch_size, shuffle=False
    )

    model = PoseModel("efficientnetv2", num_joints=33).to(device)
    criterion = HeatmapLoss()
    optimizer = create_optimizer(model, {"optimizer": "adam", "lr": args.lr})

    train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=args.epochs, device=device, run_id=1)
