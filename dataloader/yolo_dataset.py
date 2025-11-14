import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class YOLOPoseDataset(Dataset):
    def __init__(self, img_dir, label_dir, input_size=(256, 192), 
                 num_joints=33, sigma=2, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size 
        self.heatmap_size = (input_size[0]//8, input_size[1]//8)
        self.num_joints = num_joints
        self.sigma = sigma

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        self.samples = []
        failed = 0
        
        all_imgs = sorted(os.listdir(img_dir))
        all_labels = sorted(os.listdir(label_dir))
        
        for img_name, lbl_name in zip(all_imgs, all_labels):
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, lbl_name)

            if not os.path.exists(label_path):
                failed += 1
                continue

            with open(label_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) == 0:
                failed += 1
                continue
            valid = False
            for line_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5 + num_joints * 3:
                    continue

                # all zero label
                values = list(map(float, parts[1:])) 
                if all(v == 0 for v in values):
                    continue
                self.samples.append({
                    'img_name': img_name,
                    'label_line': line,
                    'person_id': line_idx
                })
                valid = True
            if not valid:
                failed += 1
                continue

        
        print(f"Loaded dataset: {len(all_imgs)} images, {len(self.samples)} person instances, {failed} failed")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample['img_name']
        label_line = sample['label_line']

        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        parts = [float(x) for x in label_line.split()]
        class_id = int(parts[0])
        
        bbox_norm = np.array(parts[1:5], dtype=np.float32)
        cx, cy, w, h = bbox_norm
        cx_abs = cx * orig_w
        cy_abs = cy * orig_h
        w_abs = w * orig_w
        h_abs = h * orig_h
        
        # Get bbox
        x1 = max(0,int(cx_abs - w_abs / 2))
        y1 = max(0, int(cy_abs - h_abs / 2))
        x2 = min(orig_w, int(cx_abs + w_abs / 2))
        y2 = min(orig_h, int(cy_abs + h_abs / 2))
        person_img = img[y1:y2, x1:x2]
        
        crop_h, crop_w = person_img.shape[:2]
        
        # Parse keypoints 
        kpts_raw = np.array(parts[5:5 + self.num_joints * 3], dtype=np.float32)
        kpts_raw = kpts_raw.reshape(self.num_joints, 3)
        kpts_xy_norm_orig = kpts_raw[:, :2]
        visibility = kpts_raw[:, 2]

        # Convert normalized keypoints to absolute coords
        kpts_xy_abs = kpts_xy_norm_orig.copy()
        kpts_xy_abs[:, 0] *= orig_w
        kpts_xy_abs[:, 1] *= orig_h
        
        # Re-coordinate keypoints relative to bbox
        kpts_xy_abs[:, 0] -= x1
        kpts_xy_abs[:, 1] -= y1
        
        # Normalize keypoints to cropped image size
        kpts_xy_norm_crop = kpts_xy_abs.copy()

        kpts_xy_norm_crop[:, 0] /= crop_w
        kpts_xy_norm_crop[:, 1] /= crop_h
        
        person_img_resized = cv2.resize(person_img, 
                                        (self.input_size[1], self.input_size[0]))  # (W, H)
        
        # Generate heatmaps and mask
        heatmaps, mask = self.kpts_to_heatmaps(
            kpts_xy_norm_crop, 
            visibility, 
            self.heatmap_size, 
            self.sigma
        )
        
        img_t = torch.from_numpy(person_img_resized).permute(2, 0, 1).float() / 255.0
        img_t = self.normalize(img_t)
        
        return {
            "image": img_t,  # (3, 256, 192)
            "heatmaps": torch.from_numpy(heatmaps),  # (33, 64, 48)
            "mask": torch.from_numpy(mask),  # (33, 1, 1)
            "kpts": torch.from_numpy(kpts_xy_norm_crop.astype(np.float32)),  # (33, 2)
            "vis": torch.from_numpy(visibility.astype(np.float32)),  # (33,)
            "name": f"{img_name}_person{sample['person_id']}"
        }
    
    def kpts_to_heatmaps(self, kpts_xy_norm, visibility, hm_size, sigma=2.0):
        K = kpts_xy_norm.shape[0]
        
        if isinstance(hm_size, int):
            Hm = Wm = hm_size
        else:
            Hm, Wm = hm_size
        
        heatmaps = np.zeros((K, Hm, Wm), dtype=np.float32)
        mask = np.zeros((K, 1, 1), dtype=np.float32)
        
        # Convert normalized coords to heatmap coords
        kpts_hm = kpts_xy_norm.copy()
        kpts_hm[:, 0] *= Wm  # kp_x
        kpts_hm[:, 1] *= Hm  # kp_y
        
        for k in range(K):
            v = visibility[k]
            if v > 0:  # visible or occluded but labeled
                cx, cy = kpts_hm[k]
                
                # Check if keypoint is within heatmap bounds
                if 0 <= cx < Wm and 0 <= cy < Hm:
                    heatmaps[k] = self.gaussian2d(Hm, Wm, cx, cy, sigma)
                    mask[k, 0, 0] = 1.0
        
        return heatmaps, mask
    
    def gaussian2d(self, H, W, cx, cy, sigma):
        # Generate 2D gaussian heatmap
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)
        y = y[:, np.newaxis]

        return np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))