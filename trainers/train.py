import torch
from tqdm import tqdm
import os
import numpy as np
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import heatmap_to_coords, compute_all_metrics
from utils.visualization import save_train_graph, save_val_graph
from models.pose_model import PoseModel
from dataloader.yolo_dataset import YOLOPoseDataset
from trainers.loss import HeatmapLoss, JointMSELoss
from trainers.optimize import create_optimizer

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=50, device="cuda", run_id=0, debug_frequency=10):
    
    os.makedirs(f"runs/train{run_id}", exist_ok=True)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_losses = []
        
        print(f"\n{'='*70}")
        print(f"🚀 EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, data in enumerate(pbar):
            imgs = data["image"].to(device)
            gt_heatmaps = data["heatmaps"].to(device)
            mask = data["mask"].to(device)
            
            # Forward pass
            pred_heatmaps = model(imgs)
            
            # Calculate loss
            loss = criterion(pred_heatmaps, gt_heatmaps, target_weight=mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # 🔍 Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print("\n❌ NaN/Inf detected in loss! Stopping training.")
                print_tensor_stats("Predictions", pred_heatmaps)
                print_tensor_stats("Targets", gt_heatmaps)
                return
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            val_loss, val_metrics = validate_model(
                model, val_loader, criterion, device, debug=(epoch == 0)
            )
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"runs/train{run_id}/best.pt")
                print(f"\n🎉 New best model! Improved by {improvement:.6f}")
        
        # 🔍 Training progress visualization
        if (epoch + 1) % 3 == 0:
            print(f"\n{'='*70}")
            print(f"📈 TRAINING PROGRESS")
            print(f"{'='*70}")
            print(f"  Epoch | Train Loss | Val Loss")
            print(f"  {'-'*40}")
            for e in range(min(5, len(train_losses))):
                val_str = f"{val_losses[e//3]:.6f}" if e % 3 == 2 and e//3 < len(val_losses) else "N/A"
                print(f"  {e+1:5d} | {train_losses[e]:10.6f} | {val_str}")

# ============================================================
# VALIDATION
# ============================================================
def validate_model(model, loader, criterion, device="cuda", debug=False):
    model.eval()
    total_loss = 0
    all_preds, all_gts, all_vis = [], [], []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            imgs = data["image"].to(device)
            gt_heatmaps = data["heatmaps"].to(device)
            mask = data["mask"].to(device)
            kpts = data["kpts"]
            vis = data["vis"]
            
            pred_heatmaps = model(imgs)
            loss = criterion(pred_heatmaps, gt_heatmaps, target_weight=mask)
            total_loss += loss.item()
            
            # 🔍 Debug first batch
            if debug and batch_idx == 0:
                print(f"\n{'='*70}")
                print("🔍 VALIDATION BATCH DEBUG")
                print(f"{'='*70}")
                print_tensor_stats("Val Predictions", pred_heatmaps)
                print_tensor_stats("Val GT", gt_heatmaps)
                print(f"Val Loss: {loss.item():.6f}")
            
            # Extract predicted coordinates
            pred_coords = heatmap_to_coords(pred_heatmaps.cpu())
            
            all_preds.append(pred_coords)
            all_gts.append(kpts.numpy())
            all_vis.append(vis.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    all_vis = np.concatenate(all_vis, axis=0)
    
    # 🔍 Coordinate statistics
    if debug:
        print(f"\n{'='*70}")
        print("📍 COORDINATE STATISTICS")
        print(f"{'='*70}")
        print(f"  Predictions shape: {all_preds.shape}")
        print(f"  GT shape: {all_gts.shape}")
        print(f"  Predictions range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
        print(f"  GT range: [{all_gts.min():.4f}, {all_gts.max():.4f}]")
        print(f"  Visible keypoints: {all_vis.sum()}/{all_vis.size}")
    
    metrics = compute_all_metrics(all_preds, all_gts)
    
    return total_loss / len(loader), metrics

# ============================================================
# ARGUMENT PARSER
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Estimation Model")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--input_size", type=int, nargs=2, default=[256, 192])
    parser.add_argument("--heatmap_size", type=int, nargs=2, default=[32, 24])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_joints", type=int, default=33)
    parser.add_argument("--sigma", type=float, default=2)
    parser.add_argument("--debug_frequency", type=int, default=10,
                        help="Print debug info every N batches")
    return parser.parse_args()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset setup
    train_img_dir = os.path.join(args.train_path, "images")
    train_label_dir = os.path.join(args.train_path, "labels")
    val_img_dir = os.path.join(args.val_path, "images")
    val_label_dir = os.path.join(args.val_path, "labels")
    
    train_dataset = YOLOPoseDataset(
        train_img_dir, train_label_dir,
        input_size=tuple(args.input_size),
        num_joints=args.num_joints,
        sigma=args.sigma
    )
    
    val_dataset = YOLOPoseDataset(
        val_img_dir, val_label_dir,
        input_size=tuple(args.input_size),
        num_joints=args.num_joints,
        sigma=args.sigma
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Model setup
    model = PoseModel("efficientnetv2", num_joints=args.num_joints).to(device)
    criterion = JointMSELoss()
    optimizer = create_optimizer(model, {"optimizer": "adam", "lr": args.lr})
    
    print(f"\n{'='*70}")
    print(f"📊 DATASET INFO")
    print(f"{'='*70}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Input size: {args.input_size}")
    print(f"  Heatmap size: {args.heatmap_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num joints: {args.num_joints}")
    print(f"  Sigma: {args.sigma}")
    
    # Start training
    train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=args.epochs, device=device, run_id=1,
        debug_frequency=args.debug_frequency
    )
    
    print(f"\n{'#'*70}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'#'*70}")