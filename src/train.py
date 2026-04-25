import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import os, sys, json, time, random, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    from torchvision.models import (
        ResNet50_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights
    )
    print(f"[OK] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("[ERROR] PyTorch not found. Install: pip install torch torchvision")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = 224
CLASSES = ["DEFECT", "PASS"]   # alphabetical = ImageFolder class order

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed()

# ── Augmentation Transforms ───────────────────────────────────────────────────

def train_transforms(img_size=IMG_SIZE):
    """
    PCB-specific rationale:
    - Rotation ±15° only  → PCBs on conveyors have small misalignment, not full spin
    - HorizontalFlip ✓    → PCBs can be loaded either side
    - VerticalFlip   ✗    → PCBs have a defined top orientation
    - ColorJitter         → Simulate real-world lighting & camera variation
    - NO heavy distortion → Corrupts solder joint geometry
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def eval_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ── Model Builders ────────────────────────────────────────────────────────────

def build_model(arch="resnet50", num_classes=2, freeze_backbone=True, dropout=0.4):
    """
    Load ImageNet pre-trained backbone → replace final layer with custom head.
    Phase 1: backbone frozen → fast head training.
    Phase 2: top layers unfrozen → domain adaptation.
    """
    arch = arch.lower()

    if arch == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_feats = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]

    elif arch == "mobilenetv2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        backbone_params = list(model.features.parameters())

    elif arch in ("efficientnetb0", "efficientnet"):
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        backbone_params = list(model.features.parameters())
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose: resnet50 | mobilenetv2 | efficientnetb0")

    if freeze_backbone:
        for p in backbone_params:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen_str = "FROZEN backbone" if freeze_backbone else "FULL fine-tune"
    print(f"[INFO] {arch} | {frozen_str} | Trainable: {trainable:,}/{total:,} params")
    return model


def unfreeze_top_layers(model, arch, n=2):
    """Selectively unfreeze last N backbone blocks for Phase 2 fine-tuning."""
    arch = arch.lower()
    if arch == "resnet50":
        children = list(model.children())[:-1]   # exclude fc
        to_unfreeze = children[-n:]
    else:
        feature_children = list(
            model.features.children() if hasattr(model, "features") else model.children()
        )
        to_unfreeze = feature_children[-n:]

    for layer in to_unfreeze:
        for p in layer.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Unfroze top {n} backbone blocks. Trainable params: {trainable:,}")
    return model

# ── Training Utilities ────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best = 0, float('inf')
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def run_epoch(model, loader, criterion, optimizer, device, scaler, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training: optimizer.zero_grad()

            if scaler and training:
                with torch.cuda.amp.autocast():
                    out = model(imgs); loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                out = model(imgs); loss = criterion(out, labels)
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def get_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)

# ── Full Training Loop ────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, cfg, device, out_dir, run_name):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["wd"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )
    stopper = EarlyStopping(patience=cfg.get("patience", 8))
    scaler  = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","val_auc","lr"]}
    best_loss, best_wts, best_epoch = float('inf'), None, 0
    save_path = out_dir / f"{run_name}_best.pth"

    print(f"\n{'='*60}")
    print(f"  {run_name}  |  LR={cfg['lr']}  |  Epochs={cfg['epochs']}")
    print(f"{'='*60}")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device, None,   training=False)
        scheduler.step()

        vl_probs, vl_labels = get_probs(model, val_loader, device)
        try:    vl_auc = roc_auc_score(vl_labels, vl_probs)
        except: vl_auc = 0.0

        lr_now = optimizer.param_groups[0]['lr']
        for k, v in zip(
            ["train_loss","val_loss","train_acc","val_acc","val_auc","lr"],
            [tr_loss, vl_loss, tr_acc, vl_acc, vl_auc, lr_now]
        ):
            history[k].append(v)

        star = ""
        if vl_loss < best_loss:
            best_loss, best_epoch = vl_loss, epoch
            best_wts = deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": vl_loss, "val_acc": vl_acc, "val_auc": vl_auc,
                        "config": cfg}, save_path)
            star = " ← BEST ✓"

        print(
            f"Ep {epoch:03d}/{cfg['epochs']} | "
            f"TrLoss {tr_loss:.4f} TrAcc {tr_acc:.3f} | "
            f"VlLoss {vl_loss:.4f} VlAcc {vl_acc:.3f} AUC {vl_auc:.3f} | "
            f"LR {lr_now:.1e} | {time.time()-t0:.1f}s{star}"
        )

        if stopper(vl_loss):
            print(f"[EARLY STOP] patience={stopper.patience} exhausted at epoch {epoch}")
            break

    model.load_state_dict(best_wts)
    history.update({"best_epoch": best_epoch, "best_val_loss": best_loss})
    print(f"\n[OK] Best: epoch {best_epoch} | val_loss {best_loss:.4f} → {save_path}")
    return history, model

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curves(history, out_dir, run_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    best_e = history.get("best_epoch", len(epochs))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"VisionSpec QC — Week 2 Learning Curves: {run_name}\n"
        "Val loss should track Train loss — widening gap = overfitting",
        fontsize=11, fontweight='bold'
    )

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], 'b-o', ms=3, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   'r--o', ms=3, label="Val Loss")
    ax.axvline(best_e, color='green', linestyle=':', label=f"Best Ep {best_e}")
    ax.set_title("Loss Curves", fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [a*100 for a in history["train_acc"]], 'b-o', ms=3, label="Train Acc%")
    ax.plot(epochs, [a*100 for a in history["val_acc"]],   'r--o', ms=3, label="Val Acc%")
    ax.axhline(90, color='orange', linestyle=':', label="Target 90%")
    ax.axvline(best_e, color='green', linestyle=':')
    ax.set_ylim(40, 105)
    ax.set_title("Accuracy Curves", fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(); ax.grid(alpha=0.3)

    # AUC
    ax = axes[2]
    ax.plot(epochs, history["val_auc"], 'g-o', ms=3, label="Val AUC-ROC")
    ax.axhline(0.95, color='orange', linestyle=':', label="Target 0.95")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Validation AUC-ROC", fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"{run_name}_learning_curves.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[OK] Learning curves → {p}")


def evaluate_model(model, test_loader, device, out_dir, run_name):
    probs, labels = get_probs(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    try:    auc = roc_auc_score(labels, probs)
    except: auc = 0.0
    acc = (preds == labels).mean()
    ap  = average_precision_score(labels, probs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"VisionSpec QC — Week 2 Test Evaluation: {run_name}\n"
        f"Test Acc: {acc*100:.2f}%  |  AUC-ROC: {auc:.4f}  |  Avg Precision: {ap:.4f}",
        fontsize=11, fontweight='bold'
    )

    # Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(labels, preds)
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticks([0,1]); ax.set_yticklabels(CLASSES)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i,j] > thresh else 'black')
    ax.set_title("Confusion Matrix (Test Set)", fontweight='bold')
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    # ROC
    ax = axes[1]
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, 'b-', lw=2, label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],'k--', alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.1, color='blue')
    ax.set_title("ROC Curve", fontweight='bold')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(); ax.grid(alpha=0.3)

    # PR Curve
    ax = axes[2]
    precision, recall, _ = precision_recall_curve(labels, probs)
    ax.plot(recall, precision, 'r-', lw=2, label=f"AP={ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color='red')
    ax.set_title("Precision-Recall Curve", fontweight='bold')
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / f"{run_name}_test_evaluation.png"
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[OK] Evaluation plots → {p}")

    report = classification_report(labels, preds, target_names=CLASSES, digits=4)
    print("\n" + "="*55 + "\nCLASSIFICATION REPORT (Test Set)\n" + "="*55)
    print(report)

    rp = out_dir / f"{run_name}_test_report.txt"
    rp.write_text(
        f"VisionSpec QC — {run_name}\n"
        f"Test Accuracy : {acc*100:.2f}%\n"
        f"AUC-ROC       : {auc:.4f}\n"
        f"Avg Precision : {ap:.4f}\n\n"
        + report + f"\nConfusion Matrix:\n{cm}"
    )
    print(f"[OK] Report → {rp}")
    return {"test_acc": acc, "test_auc": auc, "avg_precision": ap, "cm": cm.tolist()}

# ── Entry Point ───────────────────────────────────────────────────────────────

def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[INFO] Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    data = Path(args.data_dir)
    train_ds = datasets.ImageFolder(str(data/"train"), transform=train_transforms(args.img_size))
    val_ds   = datasets.ImageFolder(str(data/"val"),   transform=eval_transforms(args.img_size))
    test_ds  = datasets.ImageFolder(str(data/"test"),  transform=eval_transforms(args.img_size))

    print(f"\n[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[INFO] Classes: {train_ds.classes}")

    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  shuffle=False, **kw)

    phase = args.phase.lower()
    run_name = f"{args.model}_{phase}_{datetime.now().strftime('%m%d_%H%M')}"

    if phase == "head":
        model = build_model(args.model, freeze_backbone=True, dropout=0.4)
        cfg = {"lr": 1e-3, "wd": 1e-4, "epochs": args.epochs, "patience": 8}

    elif phase == "finetune":
        if not args.weights:
            print("[ERROR] --weights path required for finetune phase"); sys.exit(1)
        model = build_model(args.model, freeze_backbone=True, dropout=0.3)
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = unfreeze_top_layers(model, args.model, n=2)
        cfg = {"lr": 1e-4, "wd": 1e-4, "epochs": args.epochs, "patience": 10}
    else:
        print(f"[ERROR] Unknown phase: {phase}"); sys.exit(1)

    model = model.to(device)
    history, trained_model = train(model, train_dl, val_dl, cfg, device, out_dir, run_name)
    plot_learning_curves(history, out_dir, run_name)

    print("\n[INFO] Final evaluation on TEST set...")
    metrics = evaluate_model(trained_model, test_dl, device, out_dir, run_name)

    meta = {
        "run_name": run_name, "arch": args.model, "phase": phase,
        "best_weights": str(out_dir / f"{run_name}_best.pth"),
        "epochs_run": len(history["train_loss"]),
        "best_epoch": history["best_epoch"],
        **metrics
    }
    (out_dir / f"{run_name}_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{'='*60}")
    print(f"  WEEK 2 DONE | {args.model.upper()} | Phase: {phase}")
    print(f"  Test Acc  : {metrics['test_acc']*100:.2f}%")
    print(f"  AUC-ROC   : {metrics['test_auc']:.4f}")
    print(f"  Best .pth : {out_dir}/{run_name}_best.pth")
    print(f"  Next → week3_gradcam.py --weights <best.pth>")
    print(f"{'='*60}\n")


def main():
    p = argparse.ArgumentParser(description="VisionSpec QC — Week 2 Training")
    p.add_argument("--data_dir",    default="./datasets")
    p.add_argument("--model",       default="resnet50",
                   choices=["resnet50","mobilenetv2","efficientnetb0"])
    p.add_argument("--phase",       default="head", choices=["head","finetune"])
    p.add_argument("--weights",     default=None,
                   help="Path to Phase 1 checkpoint (needed for --phase finetune)")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch",       type=int, default=32)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--device",      default="auto")
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--output_dir",  default="./outputs/logs/week2")
    p.add_argument("--compare_all", action="store_true",
                   help="Quick 15-epoch run on all 3 architectures for comparison")
    args = p.parse_args()

    if args.compare_all:
        for arch in ["resnet50","mobilenetv2","efficientnetb0"]:
            args.model, args.phase, args.epochs = arch, "head", 15
            run_pipeline(args)
    else:
        run_pipeline(args)

if __name__ == "__main__":
    main()