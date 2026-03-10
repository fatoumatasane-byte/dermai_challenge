"""
baseline_model.py
=================
Starter baseline model for the DermAI Challenge.
Participants can use this as a starting point.

Usage:
    python baseline/baseline_model.py --data_dir ./data --epochs 10
    python baseline/baseline_model.py --data_dir ./data --epochs 10 --model mobilenet
"""

import argparse
import csv
import os
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from PIL import Image
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("❌ PyTorch not installed. Run: pip install torch torchvision pillow")
    sys.exit(1)

import numpy as np

# ── Dataset ────────────────────────────────────────────────────────────────────

class SkinDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, is_test=False):
        self.root_dir  = root_dir
        self.transform = transform
        self.is_test   = is_test
        self.samples   = []

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row["label"]) if "label" in row and not is_test else -1
                self.samples.append((row["image_id"], label))

    def _find_image(self, fname):
        for sub in ["", "benign", "malignant"]:
            path = os.path.join(self.root_dir, sub, fname) if sub else os.path.join(self.root_dir, fname)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image not found: {fname}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        image = Image.open(self._find_image(fname)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, fname


# ── Transforms ────────────────────────────────────────────────────────────────

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(model_name, num_classes=2):
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: resnet18, mobilenet")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            total_loss += criterion(out, labels).item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds = SkinDataset(
        os.path.join(args.data_dir, "train"),
        os.path.join(args.data_dir, "train_labels.csv"),
        transform=train_transform,
    )
    test_ds = SkinDataset(
        os.path.join(args.data_dir, "test"),
        os.path.join(args.data_dir, "test_labels_hidden.csv"),
        transform=val_transform,
        is_test=True,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} images | Test: {len(test_ds)} images")

    # Model
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}")
    print("-" * 30)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.2%}")

    # Generate submission
    model.eval()
    rows = []
    with torch.no_grad():
        for imgs, _, fnames in test_loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            for fname, pred in zip(fnames, preds):
                rows.append((fname, int(pred)))

    out_path = os.path.join(args.data_dir, "my_submission.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        for fname, pred in rows:
            writer.writerow([fname, pred])
    print(f"\n✅ Submission saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model",   default="resnet18", choices=["resnet18", "mobilenet"])
    parser.add_argument("--epochs",  type=int, default=10)
    main(parser.parse_args())
