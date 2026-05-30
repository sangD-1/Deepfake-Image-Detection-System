# ================================================================
#  Real vs AI-Generated Image Detector
#  Model   : EfficientNet-B4 (Fine-tuned)
#  Real    : CIFAKE dataset  (real/ folder)
#  Fake    : SynthBuster     (dalle2, dalle3, firefly, glide,
#            midjourney-v5, stable-diffusion-*)
#  GPU     : RTX 2050 4GB  →  batch_size=8
# ================================================================

import os, json, random, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# 0.  SEED
# ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────────────────────
# 1.  CONFIG  ← Sirf ye 3 paths apne according change karo
# ─────────────────────────────────────────────────────────────
CONFIG = {
"COCO20217_REAL_DIR": r"D:\Deepfake_project_3.0\COCO2017\train2017",
"SYNTHBUSTER_DIR"  : r"D:\Deepfake_project_3.0\synthbuster",
"OUTPUT_DIR"       : r"D:\Deepfake_project_3.0\output",

    # ── Hyperparameters (RTX 2050 4GB ke liye tuned) ──────────
    "IMG_SIZE"         : 224,
    "BATCH_SIZE"       : 8,       # 4GB VRAM ke liye safe
    "EPOCHS"           : 20,
    "LR_HEAD"          : 1e-3,    # Pehle sirf classifier head train hoga
    "LR_FULL"          : 1e-4,    # Baad mein poora model fine-tune
    "UNFREEZE_EPOCH"   : 5,       # Epoch 5 ke baad backbone unfreeze
    "WEIGHT_DECAY"     : 1e-4,
    "NUM_WORKERS"      : 0,       # Windows pe 0 rakho
    "VAL_SPLIT"        : 0.15,
    "TEST_SPLIT"       : 0.10,
    "MAX_REAL"         : 10000,   # COCO2017 se 10k real (balanced)
    "MAX_FAKE"         : 10000,   # SynthBuster se 10k fake (balanced)
}

FAKE_SUBFOLDERS = [
    "dalle2", "dalle3", "firefly", "glide", "midjourney-v5",
    "stable-diffusion-1-3", "stable-diffusion-1-4",
    "stable-diffusion-2", "stable-diffusion-xl"
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*55}")
print(f"  Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM   : {vram:.1f} GB")
print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────
# 2.  COLLECT PATHS
# ─────────────────────────────────────────────────────────────
def collect_paths(cfg):
    real_paths, fake_paths = [], []

    # ── Real images from CIFAKE ──────────────────────────────
    real_dir = Path(cfg["COCO20217_REAL_DIR"])
    if not real_dir.exists():
        raise FileNotFoundError(f"CIFAKE real dir not found:\n  {real_dir}")

    for p in real_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            real_paths.append(str(p))

    # ── Fake images from SynthBuster ────────────────────────
    fake_dir = Path(cfg["SYNTHBUSTER_DIR"])
    print("  SynthBuster subfolder scan:")
    for sub in FAKE_SUBFOLDERS:
        sub_path = fake_dir / sub
        found = []
        if sub_path.exists():
            for p in sub_path.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    found.append(str(p))
            print(f"    [{sub:<28}] {len(found):>5} images")
        else:
            print(f"    [{sub:<28}] ⚠  not found, skip")
        fake_paths.extend(found)

    # ── Balance dataset ─────────────────────────────────────
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    real_paths = real_paths[:cfg["MAX_REAL"]]
    fake_paths = fake_paths[:cfg["MAX_FAKE"]]

    print(f"\n  Real images used : {len(real_paths)}")
    print(f"  Fake images used : {len(fake_paths)}")
    print(f"  Total            : {len(real_paths)+len(fake_paths)}\n")

    all_paths  = real_paths + fake_paths
    all_labels = [0]*len(real_paths) + [1]*len(fake_paths)
    return all_paths, all_labels


# ─────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────
class RealFakeDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────────────────────
# 4.  TRANSFORMS
# ─────────────────────────────────────────────────────────────
def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────────────────────
# 5.  MODEL
# ─────────────────────────────────────────────────────────────
def build_model(freeze_backbone=True):
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

    # Replace classifier head → binary output
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1)          # Sigmoid applied in loss
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters — Total: {total/1e6:.1f}M | Trainable: {train/1e6:.1f}M")
    return model.to(DEVICE)


def unfreeze_model(model, optimizer, lr_full, weight_decay):
    """Unfreeze backbone at UNFREEZE_EPOCH"""
    for param in model.parameters():
        param.requires_grad = True
    optimizer.param_groups.clear()
    optimizer.add_param_group({
        "params": model.parameters(),
        "lr": lr_full,
        "weight_decay": weight_decay
    })
    print(f"\n  ✅ Backbone unfrozen — LR set to {lr_full}")


# ─────────────────────────────────────────────────────────────
# 6.  TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct    += (preds == labels.long()).sum().item()
        total      += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.float().to(DEVICE)

        with torch.amp.autocast("cuda"):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long()
        correct    += (preds == labels.long()).sum().item()
        total      += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


# ─────────────────────────────────────────────────────────────
# 7.  PLOT HELPERS
# ─────────────────────────────────────────────────────────────
def plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["val_auc"], color="purple", label="Val AUC")
    axes[2].set_title("ROC-AUC"); axes[2].legend(); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"), dpi=150)
    plt.close()
    print("  📊 training_history.png saved")


def plot_confusion(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  📊 confusion_matrix.png saved")


def plot_roc(y_true, y_probs, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()
    print("  📊 roc_curve.png saved")


# ─────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    out_dir = Path(cfg["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────
    print("📂 Collecting image paths...")
    all_paths, all_labels = collect_paths(cfg)

    # Train / Val / Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        all_paths, all_labels,
        test_size=cfg["TEST_SPLIT"], stratify=all_labels, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=cfg["VAL_SPLIT"]/(1-cfg["TEST_SPLIT"]),
        stratify=y_train, random_state=SEED)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_tf, val_tf = get_transforms(cfg["IMG_SIZE"])

    train_ds = RealFakeDataset(X_train, y_train, train_tf)
    val_ds   = RealFakeDataset(X_val,   y_val,   val_tf)
    test_ds  = RealFakeDataset(X_test,  y_test,  val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"],
                              shuffle=True,  num_workers=cfg["NUM_WORKERS"],
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["BATCH_SIZE"],
                              shuffle=False, num_workers=cfg["NUM_WORKERS"],
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["BATCH_SIZE"],
                              shuffle=False, num_workers=cfg["NUM_WORKERS"],
                              pin_memory=True)

    # ── Model ───────────────────────────────────────────────
    print("\n🧠 Building EfficientNet-B4...")
    model     = build_model(freeze_backbone=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["LR_HEAD"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["EPOCHS"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ───────────────────────────────────────
    history = {"train_loss":[], "val_loss":[],
               "train_acc":[],  "val_acc":[], "val_auc":[]}
    best_auc   = 0.0
    best_path  = str(out_dir / "best_model.pth")

    print(f"\n🚀 Training for {cfg['EPOCHS']} epochs...\n")
    for epoch in range(1, cfg["EPOCHS"] + 1):
        t0 = time.time()

        # Unfreeze backbone at UNFREEZE_EPOCH
        if epoch == cfg["UNFREEZE_EPOCH"]:
            unfreeze_model(model, optimizer, cfg["LR_FULL"], cfg["WEIGHT_DECAY"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg["EPOCHS"] - cfg["UNFREEZE_EPOCH"] + 1,
                eta_min=1e-6)

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc, vl_auc = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        print(f"  Epoch [{epoch:02d}/{cfg['EPOCHS']}]  "
              f"Loss: {tr_loss:.4f}/{vl_loss:.4f}  "
              f"Acc: {tr_acc:.4f}/{vl_acc:.4f}  "
              f"AUC: {vl_auc:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}  "
              f"⏱ {elapsed:.1f}s")

        # Save best model
        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_auc"    : vl_auc,
                "val_acc"    : vl_acc,
                "config"     : cfg,
            }, best_path)
            print(f"  ✅ Best model saved  (AUC={vl_auc:.4f})")

    # ── Save history ────────────────────────────────────────
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, out_dir)

    # ── Test evaluation ─────────────────────────────────────
    print(f"\n📋 Loading best model for final test evaluation...")
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    model.eval()
    all_preds, all_probs_test, all_true = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  Test"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast("cuda"):
                logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs_test.extend(probs)
            all_true.extend(labels.numpy())

    print("\n" + "="*55)
    print("  FINAL TEST RESULTS")
    print("="*55)
    print(classification_report(all_true, all_preds,
                                target_names=["Real", "Fake"]))
    test_auc = roc_auc_score(all_true, all_probs_test)
    print(f"  ROC-AUC : {test_auc:.4f}")
    print("="*55)

    plot_confusion(all_true, all_preds,    out_dir)
    plot_roc(all_true, all_probs_test,     out_dir)

    # Save final results
    results = {
        "best_val_auc"    : best_auc,
        "test_auc"        : test_auc,
        "test_accuracy"   : float(np.mean(np.array(all_preds) == np.array(all_true))),
        "best_model_path" : best_path,
        "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ All outputs saved to: {out_dir}")
    print(f"   best_model.pth | training_history.png | confusion_matrix.png | roc_curve.png")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import datetime
    main()
