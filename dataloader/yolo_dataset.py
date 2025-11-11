import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class YOLOLoader(Dataset):
    def __init__(self, img_dir, label_dir, input_size=(256,192), num_joints=33, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.num_joints = num_joints
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

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
                    if len(parts) > 5:
                        # Means bbox + keypoints exist
                        if (len(parts) - 5) >= 3:  
                            has_kp = True
                            break

                if has_kp:
                    self.valid_samples.append((img, lbl))
                else:
                    failed += 1

            except:
                failed += 1
                continue

        print(f"Loaded dataset: {len(self.valid_samples)} samples, {failed} failed to load")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_name, label_name = self.valid_samples[idx]

        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)

        # Safe default
        keypoints = np.zeros((self.num_joints, 3), dtype=np.float32)

        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = [float(x) for x in line.strip().split()]
            if len(parts) > 5:
                raw = np.array(parts[5:], dtype=np.float32).reshape(-1, 3)
                count = min(len(raw), self.num_joints)
                keypoints[:count] = raw[:count]
                break

        img = self.transform(img)
        return img, torch.tensor(keypoints, dtype=torch.float32)

