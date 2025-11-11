import torch
from tqdm import tqdm

def validate(model, dataloader, criterion, device="cuda"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, keypoints in tqdm(dataloader, desc="Validating", leave=False):
            imgs = imgs.to(device)
            keypoints = keypoints.to(device)
            preds = model(imgs)
            loss = criterion(preds, keypoints)
            total_loss += loss.item()
    return total_loss / len(dataloader)
