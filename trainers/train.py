import torch
from tqdm import tqdm
import os
import numpy as np
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.metrics import heatmap_to_coords, compute_all_metrics
# from utils.visualization import save_train_graph, save_val_graph
from models.pose_model import PoseModel
from dataloader.yolo_dataset import YOLOPoseDataset
from trainers.loss import HeatmapLoss, JointMSELoss
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
    input_size=(256, 192),
    heatmap_size=(64, 48),
):
    os.makedirs(f"runs/train{run_id}", exist_ok=True)
    
    train_log = {"loss": [], "OKS": [], "PCK": [], "PCP": [], "PDJ": []}
    val_log = {"loss": [], "OKS": [], "PCK": [], "PCP": [], "PDJ": []}
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds, all_gts = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for data in pbar: 
            imgs = data["image"]
            gt_heatmaps = data["heatmaps"]
            mask = data["mask"]
            kpts = data["kpts"]
            vis = data["vis"]
            name = data["name"] 

            imgs = imgs.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            pred_heatmaps = model(imgs)
            bs, num_joints, h, w = pred_heatmaps.size()

            print("gt_heatmaps shape:", gt_heatmaps.shape)
            print("pred_heatmaps shape:", pred_heatmaps.shape)

            idx = gt_heatmaps.view(bs, num_joints, -1).argmax(dim=2)
            pred_values = pred_heatmaps.view(bs, num_joints, -1).gather(2, idx.unsqueeze(-1)).squeeze(-1)

            # print("Pred heatmaps shape:", pred_heatmaps.shape)
            # print("GT heatmaps shape:", gt_heatmaps.shape)  
            
            loss = criterion(pred_values, torch.ones_like(pred_values).to(device))


            
            print("pred_heatmaps heatmaps shape:", pred_heatmaps.shape)
            print("gt_heatmaps heatmap max value:", gt_heatmaps.max().item())
            print("gt_heatmaps heatmap min value:", gt_heatmaps.min().item())
            print("gt_heatmaps heatmap mean value:", gt_heatmaps.mean().item())

            print("pred_heatmaps heatmap max value:", pred_heatmaps.max().item())
            print("pred_heatmaps heatmap min value:", pred_heatmaps.min().item())
            print("pred_heatmaps heatmap mean value:", pred_heatmaps.mean().item())


            print("Current batch loss:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            with torch.no_grad():
                pred_coords = heatmap_to_coords(pred_heatmaps.cpu())  # (B, 33, 3)
                gt_coords = heatmap_to_coords(gt_heatmaps.cpu())  # (B, 33, 3)
            
            all_preds.append(pred_coords)
            all_gts.append(gt_coords)
            
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
        
        metrics = compute_all_metrics(all_preds, all_gts)
        
        avg_train_loss = total_loss / len(train_loader)
        train_log["loss"].append(avg_train_loss)
        
        for k, v in metrics.items():
            train_log[k].append(v)
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")
        print(f"Train Metrics: {metrics}")
        
        # ---------- VALIDATE ----------
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            val_loss, val_metrics = validate_model(
                model, val_loader, criterion, device
            )
            
            val_log["loss"].append(val_loss)
            for k, v in val_metrics.items():
                val_log[k].append(v)
            
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Val Metrics: {val_metrics}\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(), 
                    f"runs/train{run_id}/best_.pt"
                )
                print(f"✅ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save metrics graphs
        # save_train_graph(train_log, f"runs/train{run_id}/train_metrics.png")
        # if len(val_log["loss"]) > 0:
        #     save_val_graph(val_log, f"runs/train{run_id}/val_metrics.png")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_log': train_log,
                'val_log': val_log,
            }, f"runs/train{run_id}/checkpoint_epoch{epoch+1}.pth")
    
    return train_log, val_log


# ------------------------ VALIDATION ------------------------
def validate_model(model, loader, criterion, device="cuda"):
    model.eval()
    total_loss = 0
    all_preds, all_gts = [], []
    
    with torch.no_grad():
        for imgs, gt_heatmaps in loader:
            imgs = imgs.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            pred_heatmaps = model(imgs)
            loss = criterion(pred_heatmaps, gt_heatmaps)
            total_loss += loss.item()
            
            pred_coords = heatmap_to_coords(pred_heatmaps.cpu())
            gt_coords = heatmap_to_coords(gt_heatmaps.cpu())
            
            all_preds.append(pred_coords)
            all_gts.append(gt_coords)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    metrics = compute_all_metrics(all_preds, all_gts)
    
    return total_loss / len(loader), metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Estimation Model")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training data folder (contains 'images' and 'labels')")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to validation data folder (contains 'images' and 'labels')")
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 192],
                        help="Input image size (height width)")
    parser.add_argument("--heatmap_size", type=int, nargs=2, default=[64, 48], 
                        help="Heatmap output size (height width)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_joints", type=int, default=33,
                        help="Number of keypoints")
    parser.add_argument("--sigma", type=float, default=2,
                        help="Gaussian sigma for heatmap generation")
    return parser.parse_args()


# -------------- MAIN ----------------
if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    train_img_dir = os.path.join(args.train_path, "images")
    train_label_dir = os.path.join(args.train_path, "labels")
    val_img_dir = os.path.join(args.val_path, "images")
    val_label_dir = os.path.join(args.val_path, "labels")
    train_dataset = YOLOPoseDataset(
        train_img_dir, 
        train_label_dir, 
        input_size=tuple(args.input_size),
        num_joints=args.num_joints,
        sigma=args.sigma
    )
    
    val_dataset = YOLOPoseDataset(
        val_img_dir, 
        val_label_dir, 
        input_size=tuple(args.input_size),
        num_joints=args.num_joints,
        sigma=args.sigma
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,  
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = PoseModel(
        "efficientnetv2", 
        num_joints=args.num_joints
    ).to(device)
    criterion = HeatmapLoss()
    
    optimizer = create_optimizer(model, {"optimizer": "adam", "lr": args.lr})
    
    print(f"\n📊 Dataset Info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Input size: {args.input_size}")
    print(f"  Batch size: {args.batch_size}\n")
    train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        num_epochs=args.epochs, 
        device=device, 
        run_id=1,
        input_size=tuple(args.input_size),
        heatmap_size=tuple(args.heatmap_size)
    )
    
    print("---------------------------------------------------------------  Training completed!")