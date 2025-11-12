import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

def generate_heatmap(keypoints, heatmap_size, sigma=2):
    """
    Generate Gaussian heatmaps for keypoints.
    
    Args:
        keypoints: (num_joints, 3) - [x, y, visibility]
        heatmap_size: (H, W) - target heatmap dimensions
        sigma: Gaussian kernel standard deviation
    
    Returns:
        heatmaps: (num_joints, H, W)
    """
    num_joints = keypoints.shape[0]
    H, W = heatmap_size
    heatmaps = np.zeros((num_joints, H, W), dtype=np.float32)
    
    # Gaussian kernel size
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = size // 2, size // 2
    
    for idx in range(num_joints):
        x_coord, y_coord, vis = keypoints[idx]
        
        # Skip invisible keypoints
        if vis < 0.5:
            continue
        
        # Convert to heatmap coordinates
        mu_x = int(x_coord * W)
        mu_y = int(y_coord * H)
        
        # Check if keypoint is within bounds
        if mu_x < 0 or mu_y < 0 or mu_x >= W or mu_y >= H:
            continue
        
        # Generate Gaussian kernel
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        
        # Determine the range to place the Gaussian
        ul = [int(mu_x - size // 2), int(mu_y - size // 2)]
        br = [int(mu_x + size // 2 + 1), int(mu_y + size // 2 + 1)]
        
        # Clip to heatmap boundaries
        g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
        
        img_x = max(0, ul[0]), min(br[0], W)
        img_y = max(0, ul[1]), min(br[1], H)
        
        # Add Gaussian to heatmap
        heatmaps[idx, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    
    return heatmaps


class YOLOPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, input_size=(256, 192), 
                 num_joints=33, sigma=2, transform=None):
        """
        Args:
            img_dir: Directory with images
            label_dir: Directory with YOLO format labels
            input_size: (H, W) for input images
            heatmap_size: (H, W) for output heatmaps (should match model output)
            num_joints: Number of keypoints
            sigma: Gaussian kernel sigma for heatmap generation
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size  # (H, W)
        # print("Input size of dataloader:", self.input_size)
        self.heatmap_size = (int(input_size[0]/8), int(input_size[1]/8))
        # print("Heatmap size of dataloader:", self.heatmap_size)  
        self.num_joints = num_joints
        self.sigma = sigma
        
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load valid samples
        all_imgs = sorted(os.listdir(img_dir))
        all_labels = sorted(os.listdir(label_dir))
        
        self.valid_samples = []
        failed = 0
        
        for img, lbl in zip(all_imgs, all_labels):
            label_path = os.path.join(label_dir, lbl)
            if not os.path.exists(label_path):
                failed += 1
                continue
            
            try:
                with open(label_path) as f:
                    lines = f.readlines()
                
                has_kp = False
                for line in lines:
                    parts = line.strip().split()
                    # YOLO format: class_id x y w h kp1_x kp1_y kp1_v ...
                    if len(parts) >= 5 + num_joints * 3:
                        has_kp = True
                        break
                
                if has_kp:
                    self.valid_samples.append((img, lbl))
                else:
                    failed += 1
            except:
                failed += 1
                continue
        
        print(f"Loaded dataset: {len(self.valid_samples)} samples, {failed} failed")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_name, label_name = self.valid_samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        input_h, input_w = self.input_size
        img = cv2.resize(img, (input_w, input_h))
        keypoints = np.zeros((self.num_joints, 3), dtype=np.float32)
        
        with open(label_path) as f:
            lines = f.readlines()
        
        for line in lines:
            parts = [float(x) for x in line.strip().split()]
            if len(parts) >= 5 + self.num_joints * 3:
                raw = np.array(parts[5:], dtype=np.float32).reshape(-1, 3)
                count = min(len(raw), self.num_joints)
                keypoints[:count] = raw[:count]
                break
        heatmaps = generate_heatmap(keypoints, self.heatmap_size, self.sigma)
        
        img = self.transform(img)
        
        return img, torch.from_numpy(heatmaps)
