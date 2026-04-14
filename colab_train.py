"""
══════════════════════════════════════════════════════════════════════
  Google Colab Training Script — Masked Autoencoder
══════════════════════════════════════════════════════════════════════

HOW TO USE (Google Colab — FREE GPU):
─────────────────────────────────────
1. Go to https://colab.research.google.com
2. Click File → New notebook
3. Click Runtime → Change runtime type → GPU (T4)
4. Upload your project folder to Colab OR clone from GitHub
5. Copy-paste this ENTIRE script into a Colab cell and run it

The script will:
  • Install dependencies
  • Train for 50 epochs on GPU (~2-3 minutes with T4)
  • Save the model to checkpoints/best_model.pth
  • Download the checkpoint for local use
"""

# ── Cell 1: Setup ────────────────────────────────────────────────────────────
# Run this cell first to install dependencies and upload your project

import subprocess, sys, os

# Install dependencies (if not already installed)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'torch', 'torchvision', 'scikit-image', 'matplotlib', 'Pillow'])

print("✅ Dependencies installed")

# Check GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Cell 2: Model & Utils (self-contained — no file imports needed) ──────────

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = 64
PATCH_SIZE = 8
EPOCHS = 50
BATCH_SIZE = 128      # GPU can handle larger batches
LR = 1e-3
MASK_RATIO = 0.5
SAVE_DIR = './checkpoints'
DATA_DIR = './data'

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Masking ──────────────────────────────────────────────────────────────────

def create_patch_mask(image_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=0.5):
    patches_per_side = image_size // patch_size
    num_patches = patches_per_side ** 2
    num_masked = int(num_patches * mask_ratio)
    perm = torch.randperm(num_patches)
    masked_indices = perm[:num_masked]
    mask = torch.ones(num_patches)
    mask[masked_indices] = 0.0
    mask = mask.reshape(patches_per_side, patches_per_side)
    mask = mask.repeat_interleave(patch_size, dim=0)
    mask = mask.repeat_interleave(patch_size, dim=1)
    return mask.unsqueeze(0), masked_indices


# ── Model ────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class MaskedAutoencoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128);          self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256);         self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.output_conv = nn.Sequential(nn.Conv2d(64, in_channels, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.output_conv(d1)


# ── Dataset ──────────────────────────────────────────────────────────────────

class MaskedImageDataset(Dataset):
    def __init__(self, base_dataset, mask_ratio=0.5):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image = item[0] if isinstance(item, (tuple, list)) else item
        mask, _ = create_patch_mask(mask_ratio=self.mask_ratio)
        return image * mask, image, mask


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

full_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(MaskedImageDataset(train_ds, MASK_RATIO),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(MaskedImageDataset(val_ds, MASK_RATIO),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"✅ Dataset loaded: {len(train_ds)} train / {len(val_ds)} val images")
print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


# ── Cell 3: Training ────────────────────────────────────────────────────────

model = MaskedAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\n{'='*70}")
print(f" Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f" Training {EPOCHS} epochs on {device}")
print(f"{'='*70}\n")

train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # Train
    model.train()
    running = 0.0
    for masked, original, mask in train_loader:
        masked, original = masked.to(device), original.to(device)
        optimizer.zero_grad()
        out = model(masked)
        loss = criterion(out, original)
        loss.backward()
        optimizer.step()
        running += loss.item()
    avg_train = running / len(train_loader)
    train_losses.append(avg_train)

    # Validate
    model.eval()
    running = 0.0
    with torch.no_grad():
        for masked, original, mask in val_loader:
            masked, original = masked.to(device), original.to(device)
            loss = criterion(model(masked), original)
            running += loss.item()
    avg_val = running / len(val_loader)
    val_losses.append(avg_val)

    scheduler.step()
    elapsed = time.time() - t0

    marker = ""
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train, 'val_loss': avg_val,
            'img_size': IMG_SIZE,
        }, os.path.join(SAVE_DIR, 'best_model.pth'))
        marker = " ★"

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  |  train {avg_train:.6f}  |  "
              f"val {avg_val:.6f}  |  {elapsed:.1f}s{marker}")

print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")


# ── Cell 4: Plot & Save ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, 'b-', lw=2, label='Train')
ax.plot(val_losses, 'r-', lw=2, label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('Training History'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'), dpi=150)
plt.show()
print(f"✅ History plot saved")


# ── Cell 5: Download checkpoint ─────────────────────────────────────────────
# In Google Colab, uncomment the lines below to download the trained model:

# from google.colab import files
# files.download('checkpoints/best_model.pth')

print(f"\n{'='*70}")
print("NEXT STEPS:")
print("  1. Download checkpoints/best_model.pth")
print("  2. Place it in your local project: checkpoints/best_model.pth")
print("  3. Run: python app.py")
print("  4. Open: http://localhost:5000")
print(f"{'='*70}")
