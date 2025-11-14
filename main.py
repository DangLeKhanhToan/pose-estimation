import os
import cv2
import numpy as np
import argparse
import torch

SKELETON = [
    (9,11),(11,13),(13,15),           # left arm: shoulder-elbow-wrist
    (10,12),(12,14),(14,16),          # right arm
    (9,10),                           # shoulders
    (15,16),(15,23),(16,24),          # hips & torso
    (23,25),(25,27),(27,29),(29,31),  # left leg chain
    (24,26),(26,28),(28,30),(30,32),  # right leg chain
    (31,21),(32,22),(21,19),(22,20),  # feet/heels/ankles (approx)
    (0,9),(0,10),                     # nose to shoulders
    (1,2),(2,3),(4,5),(5,6),          # eyes chains
    (7,9),(8,10)                      # ears to shoulders
]
NUM_JOINTS = 33
INPUT_IMAGE_SIZE = (256, 192)  # (height, width)
GAUSS_SIGMA  = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pose Estimation Model")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    return parser.parse_args()