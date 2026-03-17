# =============================================================================
# MediScan AI - Production Model Training Notebook
# Version: 5.0 | EfficientNetB4 + Transfer Learning + Grad-CAM
# Run this on Kaggle with GPU T4 x2 enabled
#
# DATASET: Add "Chest X-Ray Images (Pneumonia)" by Paul Mooney
# Path auto-mounts at: /kaggle/input/chest-xray-pneumonia/chest_xray
# =============================================================================

# ------------------------------------------------------------------------------
# CELL 1 — Imports & Configuration
# ------------------------------------------------------------------------------

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {device}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---- Paths ----
DATA_DIR    = "/kaggle/input/chest-xray-pneumonia/chest_xray"
OUTPUT_DIR  = "/kaggle/working"
MODEL_PATH  = os.path.join(OUTPUT_DIR, "mediscan_v5.pth")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")

# ---- Hyperparameters ----
CONFIG = {
    "image_size"    : 380,       # EfficientNetB4 native resolution
    "batch_size"    : 16,        # Reduced for 380px images
    "num_epochs"    : 25,
    "lr_head"       : 1e-3,      # Learning rate for classifier head
    "lr_backbone"   : 1e-4,      # Learning rate for fine-tuned backbone
    "weight_decay"  : 1e-4,
    "dropout"       : 0.4,
    "label_smoothing": 0.1,
    "early_stop_patience": 6,
    "unfreeze_epoch": 5,         # Epoch to unfreeze backbone layers
    "num_classes"   : 2,
    "classes"       : ["NORMAL", "PNEUMONIA"],
}

print("\nConfiguration loaded.")
print(json.dumps({k: v for k, v in CONFIG.items() if k != "classes"}, indent=2))


# ------------------------------------------------------------------------------
# CELL 2 — Data Pipeline
# ------------------------------------------------------------------------------

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


def make_weighted_sampler(dataset):
    """
    Handles class imbalance (PNEUMONIA >> NORMAL in this dataset)
    by oversampling the minority class during training.
    """
    class_counts = np.bincount(dataset.targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in dataset.targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


train_tf, eval_tf = get_transforms()

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=eval_tf)

sampler = make_weighted_sampler(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          sampler=sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"],
                          shuffle=False,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"],
                          shuffle=False,  num_workers=4, pin_memory=True)

print(f"Classes  : {train_dataset.classes}")
print(f"Train    : {len(train_dataset):,} images")
print(f"Val      : {len(val_dataset):,} images")
print(f"Test     : {len(test_dataset):,} images")

# Class distribution
counts = np.bincount(train_dataset.targets)
for cls, cnt in zip(train_dataset.classes, counts):
    print(f"  {cls}: {cnt:,} ({cnt/len(train_dataset)*100:.1f}%)")


# ------------------------------------------------------------------------------
# CELL 3 — Model Architecture
# ------------------------------------------------------------------------------

class MediScanModel(nn.Module):
    """
    EfficientNetB4 backbone with custom classification head.

    Training strategy:
    - Phase 1 (epochs 1-5):  Only classifier head trains. Backbone frozen.
    - Phase 2 (epochs 6-25): Last 30 backbone layers + head fine-tune together.
    
    This prevents destroying pre-trained ImageNet features early in training.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        # Load EfficientNetB4 with ImageNet weights
        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Extract feature layers (everything except the default classifier)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # Custom classifier head
        in_features = backbone.classifier[1].in_features  # 1792
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Freeze backbone initially
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
        print("Backbone frozen.")

    def unfreeze_last_n_layers(self, n: int = 30):
        """Unfreeze the last n parameter tensors of the backbone."""
        params = list(self.features.parameters())
        for param in params[-n:]:
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfrozen last {n} backbone layers. Trainable params: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = MediScanModel(
    num_classes=CONFIG["num_classes"],
    dropout=CONFIG["dropout"]
).to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ------------------------------------------------------------------------------
# CELL 4 — Training Infrastructure
# ------------------------------------------------------------------------------

def get_optimizer_and_scheduler(model, phase: str):
    if phase == "head":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CONFIG["lr_head"],
            weight_decay=CONFIG["weight_decay"]
        )
    else:  # fine-tune
        optimizer = optim.AdamW([
            {"params": model.features.parameters(), "lr": CONFIG["lr_backbone"]},
            {"params": model.classifier.parameters(), "lr": CONFIG["lr_head"]},
        ], weight_decay=CONFIG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixed precision training for faster GPU computation
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def evaluate_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), 100.0 * correct / total, auc


# Loss with label smoothing to prevent overconfident predictions
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
scaler    = torch.cuda.amp.GradScaler()
optimizer, scheduler = get_optimizer_and_scheduler(model, phase="head")

print("Training infrastructure ready.")


# ------------------------------------------------------------------------------
# CELL 5 — Main Training Loop
# ------------------------------------------------------------------------------

history = {k: [] for k in ["train_loss","train_acc","val_loss","val_acc","val_auc","lr"]}
best_val_auc   = 0.0
best_val_acc   = 0.0
patience_count = 0
start_time     = time.time()

print(f"\nStarting training for {CONFIG['num_epochs']} epochs...\n")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'AUC':>8} {'LR':>10}")
print("-" * 75)

for epoch in range(1, CONFIG["num_epochs"] + 1):

    # Switch to fine-tuning phase
    if epoch == CONFIG["unfreeze_epoch"] + 1:
        print(f"\n--- Switching to fine-tune phase at epoch {epoch} ---")
        model.unfreeze_last_n_layers(n=30)
        optimizer, scheduler = get_optimizer_and_scheduler(model, phase="finetune")
        scaler = torch.cuda.amp.GradScaler()
        print()

    train_loss, train_acc            = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
    val_loss, val_acc, val_auc       = evaluate_epoch(model, val_loader, criterion)
    current_lr                       = optimizer.param_groups[0]["lr"]
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_auc"].append(val_auc)
    history["lr"].append(current_lr)

    improved = ""
    if val_auc > best_val_auc:
        best_val_auc   = val_auc
        best_val_acc   = val_acc
        patience_count = 0
        torch.save({
            "epoch"            : epoch,
            "model_state_dict" : model.state_dict(),
            "classes"          : train_dataset.classes,
            "config"           : CONFIG,
            "val_auc"          : best_val_auc,
            "val_acc"          : best_val_acc,
        }, MODEL_PATH)
        improved = "  [BEST SAVED]"
    else:
        patience_count += 1

    print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>9.2f}% {val_loss:>10.4f} "
          f"{val_acc:>8.2f}% {val_auc:>8.4f} {current_lr:>10.2e}{improved}")

    if patience_count >= CONFIG["early_stop_patience"]:
        print(f"\nEarly stopping triggered at epoch {epoch}.")
        break

elapsed = (time.time() - start_time) / 60
print(f"\nTraining complete in {elapsed:.1f} minutes.")
print(f"Best Val AUC : {best_val_auc:.4f}")
print(f"Best Val Acc : {best_val_acc:.2f}%")

with open(HISTORY_PATH, "w") as f:
    json.dump(history, f)


# ------------------------------------------------------------------------------
# CELL 6 — Test Set Evaluation
# ------------------------------------------------------------------------------

# Load best checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

print("=" * 60)
print("FINAL TEST SET RESULTS")
print("=" * 60)
print(classification_report(
    all_labels, all_preds,
    target_names=CONFIG["classes"],
    digits=4
))
test_auc = roc_auc_score(all_labels, all_probs)
print(f"AUC-ROC   : {test_auc:.4f}")
print(f"Val AUC   : {best_val_auc:.4f}")


# ------------------------------------------------------------------------------
# CELL 7 — Diagnostic Plots
# ------------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("MediScan V5.0 — Training Report", fontsize=14, fontweight='bold')

# Loss curve
axes[0,0].plot(history["train_loss"], label="Train", linewidth=2)
axes[0,0].plot(history["val_loss"],   label="Val",   linewidth=2)
axes[0,0].set_title("Loss")
axes[0,0].set_xlabel("Epoch")
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# Accuracy curve
axes[0,1].plot(history["train_acc"], label="Train", linewidth=2)
axes[0,1].plot(history["val_acc"],   label="Val",   linewidth=2)
axes[0,1].set_title("Accuracy (%)")
axes[0,1].set_xlabel("Epoch")
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# AUC curve
axes[0,2].plot(history["val_auc"], color="green", linewidth=2)
axes[0,2].set_title("Validation AUC-ROC")
axes[0,2].set_xlabel("Epoch")
axes[0,2].grid(alpha=0.3)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,0],
            xticklabels=CONFIG["classes"], yticklabels=CONFIG["classes"],
            cmap='Blues')
axes[1,0].set_title("Confusion Matrix (Test Set)")
axes[1,0].set_ylabel("True")
axes[1,0].set_xlabel("Predicted")

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
axes[1,1].plot(fpr, tpr, linewidth=2, label=f"AUC = {test_auc:.4f}")
axes[1,1].plot([0,1],[0,1], 'k--', alpha=0.4)
axes[1,1].set_title("ROC Curve (Test Set)")
axes[1,1].set_xlabel("False Positive Rate")
axes[1,1].set_ylabel("True Positive Rate")
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
ap = average_precision_score(all_labels, all_probs)
axes[1,2].plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
axes[1,2].set_title("Precision-Recall Curve")
axes[1,2].set_xlabel("Recall")
axes[1,2].set_ylabel("Precision")
axes[1,2].legend()
axes[1,2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_report.png"), dpi=150, bbox_inches='tight')
plt.show()
print("Training report saved.")


# ------------------------------------------------------------------------------
# CELL 8 — Grad-CAM Verification (smoke test)
# ------------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Generates a heatmap showing which regions the model focused on.
    Attached to the last convolutional block of EfficientNetB4.
    """
    def __init__(self, model: nn.Module):
        self.model     = model
        self.gradients = None
        self.activations = None
        # Hook into the last features block
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        import torch.nn.functional as F
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        return cam.squeeze().cpu().numpy(), class_idx


# Quick smoke test on one test image
gradcam = GradCAM(model)
sample_img, sample_label = test_dataset[0]
input_tensor = sample_img.unsqueeze(0).to(device)
cam, pred_idx = gradcam.generate(input_tensor)
print(f"Grad-CAM smoke test passed.")
print(f"  True label : {CONFIG['classes'][sample_label]}")
print(f"  Prediction : {CONFIG['classes'][pred_idx]}")
print(f"  CAM shape  : {cam.shape}")


# ------------------------------------------------------------------------------
# CELL 9 — Export & Summary
# ------------------------------------------------------------------------------

model_size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)

print("\n" + "=" * 60)
print("EXPORT SUMMARY")
print("=" * 60)
print(f"Model file   : {MODEL_PATH}")
print(f"Model size   : {model_size_mb:.1f} MB")
print(f"History file : {HISTORY_PATH}")
print()
print("PERFORMANCE SUMMARY")
print(f"  Best Val AUC  : {best_val_auc:.4f}")
print(f"  Best Val Acc  : {best_val_acc:.2f}%")
print(f"  Test AUC      : {test_auc:.4f}")
print()
print("FILES TO DOWNLOAD (Kaggle sidebar -> Output):")
print("  1. mediscan_v5.pth          <- Model weights")
print("  2. training_history.json    <- Training metrics")
print("  3. training_report.png      <- Diagnostic plots")
print()
print("Place mediscan_v5.pth at:")
print("  mediscan-v5/model/mediscan_v5.pth")
