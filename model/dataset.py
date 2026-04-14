"""
Dataset module — CIFAR-10 and custom image-folder loaders.

Both loaders return (masked_image, original_image, mask) triples
via the `MaskedImageDataset` wrapper.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

from .utils import IMG_SIZE, create_patch_mask, apply_mask


# ─── Masked wrapper ─────────────────────────────────────────────────────────

class MaskedImageDataset(Dataset):
    """
    Wraps any image dataset and adds random patch masking on-the-fly.

    Each call to __getitem__ returns:
        masked_image : Tensor (C, H, W)  — image with some patches zeroed
        original     : Tensor (C, H, W)  — clean target
        mask         : Tensor (1, H, W)  — binary mask (1 = visible)
    """

    def __init__(self, base_dataset, mask_ratio: float = 0.5):
        self.base_dataset = base_dataset
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image = item[0] if isinstance(item, (tuple, list)) else item

        mask, _ = create_patch_mask(mask_ratio=self.mask_ratio)
        masked_image = apply_mask(image, mask)

        return masked_image, image, mask


# ─── CIFAR-10 loader ────────────────────────────────────────────────────────

def get_cifar10_dataset(data_dir: str = './data',
                        mask_ratio: float = 0.5,
                        batch_size: int = 64,
                        subset_size: int = 0):
    """
    Download CIFAR-10, resize to IMG_SIZE, and return masked data loaders.

    Args:
        data_dir    : Where to cache the raw CIFAR-10 files.
        mask_ratio  : Fraction of patches to mask (0.0–1.0).
        batch_size  : Mini-batch size.
        subset_size : If > 0, use only this many images (for fast CPU training).
                      Set to 0 to use the full dataset.

    Returns:
        train_loader, val_loader  (DataLoader instances)
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Optionally take a small subset for fast CPU training
    if subset_size > 0 and subset_size < len(full_dataset):
        full_dataset = torch.utils.data.Subset(
            full_dataset,
            torch.randperm(len(full_dataset),
                           generator=torch.Generator().manual_seed(42))[:subset_size].tolist()
        )
        print(f"[INFO] Using subset: {subset_size} images")

    # 90 / 10 train-val split
    total = len(full_dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        MaskedImageDataset(train_ds, mask_ratio),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # 0 is safest on Windows
        pin_memory=False,
    )
    val_loader = DataLoader(
        MaskedImageDataset(val_ds, mask_ratio),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader


# ─── Custom image-folder loader ─────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """
    Load all images from a flat directory.

    Supported extensions: .jpg .jpeg .png .bmp .tiff .webp
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        self.image_paths = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)
