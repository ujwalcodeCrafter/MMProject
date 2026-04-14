"""
Training script for the Masked Autoencoder.

Usage:
    python model/train.py                        # Default: 50 epochs
    python model/train.py --epochs 100 --lr 5e-4
    python model/train.py --quick                # Fast CPU: 10 epochs, 5K images (~3-5 min)
    python model/train.py --subset 5000          # Use only 5000 images
    python model/train.py --mask_ratio 0.75      # 75 % masking
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the project root is on sys.path so `model.*` imports work
# even when this file is invoked as `python model/train.py`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.model import MaskedAutoencoder
from model.dataset import get_cifar10_dataset
from model.utils import IMG_SIZE


# ─── Training loop ──────────────────────────────────────────────────────────

def train(args):
    """Full training procedure: data → model → optimise → save."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] Image size        : {IMG_SIZE}×{IMG_SIZE}")
    print(f"[INFO] Mask ratio        : {args.mask_ratio}")
    print(f"[INFO] Epochs            : {args.epochs}")
    print(f"[INFO] Learning rate     : {args.lr}")
    print(f"[INFO] Save directory    : {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    print(f"[INFO] Subset size       : {'full' if args.subset == 0 else args.subset}")

    print("\n[INFO] Downloading / loading CIFAR-10 …")
    train_loader, val_loader = get_cifar10_dataset(
        data_dir=args.data_dir,
        mask_ratio=args.mask_ratio,
        batch_size=args.batch_size,
        subset_size=args.subset,
    )
    print(f"[INFO] Train batches : {len(train_loader)}")
    print(f"[INFO] Val   batches : {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────
    model = MaskedAutoencoder(in_channels=3).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters    : {num_params:,}")

    # ── Optimisation ─────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    l1_loss = nn.L1Loss()  # Add L1 for sharper edges
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── History ──────────────────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float('inf')

    print(f"\n{'='*70}")
    print(f" {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}  {'Time':>7}")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for masked_imgs, original_imgs, masks in train_loader:
            masked_imgs = masked_imgs.to(device)
            original_imgs = original_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(masked_imgs)
            
            # Combined loss: MSE + L1 for sharper reconstructions
            mse = criterion(outputs, original_imgs)
            l1 = l1_loss(outputs, original_imgs)
            loss = 0.7 * mse + 0.3 * l1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for masked_imgs, original_imgs, masks in val_loader:
                masked_imgs = masked_imgs.to(device)
                original_imgs = original_imgs.to(device)
                outputs = model(masked_imgs)
                
                # Combined loss same as training
                mse = criterion(outputs, original_imgs)
                l1 = l1_loss(outputs, original_imgs)
                loss = 0.7 * mse + 0.3 * l1
                running_loss += loss.item()

        avg_val = running_loss / len(val_loader)
        val_losses.append(avg_val)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        marker = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
                'img_size': IMG_SIZE,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            marker = "  ★ best"

        print(f" {epoch:5d}  {avg_train:12.6f}  {avg_val:12.6f}"
              f"  {lr_now:10.6f}  {elapsed:5.1f}s{marker}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
                'img_size': IMG_SIZE,
            }, path)

    print(f"{'='*70}")
    print(f"[INFO] Training complete — best val loss: {best_val_loss:.6f}")

    # ── Loss plot ────────────────────────────────────────────────────────
    _plot_history(train_losses, val_losses, args.save_dir)

    return model


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _plot_history(train_losses, val_losses, save_dir):
    """Save a training / validation loss curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', lw=2, label='Train Loss')
    ax.plot(epochs, val_losses,   'r-', lw=2, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('MSE Loss', fontsize=13)
    ax.set_title('Training History — Masked Autoencoder', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, 'training_history.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Loss plot saved → {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the Masked Autoencoder on CIFAR-10')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--mask_ratio', type=float, default=0.35)  # Reduced masking ratio for easier learning
    parser.add_argument('--data_dir',   type=str,   default='./data')
    parser.add_argument('--save_dir',   type=str,   default='./checkpoints')
    parser.add_argument('--subset',     type=int,   default=0,
                        help='Use only N images (0 = full dataset)')
    parser.add_argument('--quick',      action='store_true',
                        help='Fast CPU run: 10 epochs, 5000 images')

    args = parser.parse_args()
    if args.quick:
        args.epochs = 10
        if args.subset == 0:
            args.subset = 5000
        print(f"[INFO] Quick mode → {args.epochs} epochs, {args.subset} images\n")

    train(args)
