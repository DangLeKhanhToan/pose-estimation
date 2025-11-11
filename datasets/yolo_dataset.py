import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class YOLOLoader(Dataset):
    def __init__(self, img_dir, label_dir, input_size=(256,192), num_joints=17, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.input_size = input_size
        self.num_joints = num_joints
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)

        if os.path.exists(label_path):
            with open(label_path) as f:
                lines = f.readlines()
            keypoints = []
            for line in lines:
                parts = line.strip().split()
                # Format: class cx cy w h (optional keypoints)
                if len(parts) >= 5:
                    parts = [float(x) for x in parts]
                    # If keypoints exist after bbox (x1 y1 x2 y2 ...)
                    if len(parts) > 5:
                        keypoints = np.array(parts[5:]).reshape(-1, 3)
            keypoints = np.array(keypoints, dtype=np.float32)
        else:
            keypoints = np.zeros((self.num_joints, 3), dtype=np.float32)

        img = self.transform(img)
        return img, torch.tensor(keypoints, dtype=torch.float32)
