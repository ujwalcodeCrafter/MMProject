# Self-Supervised Mask-Based Image Reconstruction with Consistency Analysis

A complete end-to-end deep learning project that uses a **self-supervised Masked Autoencoder** to reconstruct hidden regions of images and analyse reconstruction quality via PSNR, SSIM, and pixel-wise error heatmaps.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)

---

## Features

| Feature | Description |
|---------|-------------|
| **Masked Autoencoder** | UNet-style CNN with skip connections (≈ 7 M params) |
| **Self-Supervised Training** | Trained on CIFAR-10 — no manual labels needed |
| **Configurable Masking** | Random patch masking with adjustable ratio (10 %–90 %) |
| **Consistency Analysis** | PSNR, SSIM, and per-pixel error heatmap |
| **Web Interface** | Drag-and-drop upload, live results, dark-themed UI |

---

## Project Structure

```
project/
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── model/
│   ├── __init__.py
│   ├── model.py            # UNet Masked Autoencoder architecture
│   ├── dataset.py          # CIFAR-10 + custom folder dataset loaders
│   ├── train.py            # Training script with CLI
│   └── utils.py            # Masking, metrics (PSNR/SSIM), visualisation
│
├── templates/
│   └── index.html          # Web UI
│
├── static/
│   ├── uploads/            # User-uploaded images
│   └── outputs/            # Generated result images
│
├── checkpoints/            # Saved model weights (created during training)
└── data/                   # CIFAR-10 cache (created during training)
```

---

## Quick Start

### 1. Install Dependencies

```bash
# (Recommended) Create a virtual environment first
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Train the Model

**Quick test (10 epochs, ~2-5 min on GPU):**
```bash
python model/train.py --quick
```

**Full training (50 epochs, ~15-30 min on GPU):**
```bash
python model/train.py --epochs 50 --mask_ratio 0.5
```

**All training options:**
```
--epochs      Number of training epochs (default: 50)
--batch_size  Batch size (default: 64)
--lr          Learning rate (default: 0.001)
--mask_ratio  Fraction of patches to mask (default: 0.5)
--data_dir    Dataset cache directory (default: ./data)
--save_dir    Checkpoint directory (default: ./checkpoints)
--quick       Quick 10-epoch run
```

The first run automatically downloads CIFAR-10 (~170 MB).
Training saves:
- `checkpoints/best_model.pth` — best validation loss
- `checkpoints/training_history.png` — loss curves

### 3. Run the Web Application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 4. Use the Web App

1. **Upload** any image (drag-and-drop or click to browse)
2. **Adjust** the mask ratio slider (10 %–90 %)
3. Click **Reconstruct Image**
4. View results:
   - Original image (resized to 64×64)
   - Mask visualization (red overlay on masked patches)
   - Masked input (what the model sees)
   - Reconstructed image (model output)
   - Error heatmap (per-pixel absolute error)
   - PSNR and SSIM metrics

---

## How It Works

### Self-Supervised Learning Pipeline

```
Input Image  →  Random Patch Masking  →  Masked Image  →  UNet Model  →  Reconstructed Image
      ↑                                                                         ↓
      └──────────────── MSE Loss ←──────────────────────────────────────────────┘
```

1. **Masking**: The image is divided into 8×8 pixel patches. A configurable fraction is randomly zeroed out.
2. **Reconstruction**: The UNet autoencoder receives the masked image and predicts the full image.
3. **Training signal**: MSE loss between the model output and the original (unmasked) image. No labels required — this is fully self-supervised.

### Model Architecture

```
Encoder                          Decoder
───────                          ───────
Conv(3→64) ─────────────────── → Conv(128→64) → Output(64→3, Sigmoid)
  ↓ MaxPool                            ↑ UpConv
Conv(64→128) ───────────────── → Conv(256→128)
  ↓ MaxPool                            ↑ UpConv
Conv(128→256) ─────────────── → Conv(512→256)
  ↓ MaxPool                            ↑ UpConv
        Bottleneck: Conv(256→512)
```

Skip connections between corresponding encoder/decoder stages preserve spatial detail.

### Consistency Analysis

| Metric | What It Measures |
|--------|-----------------|
| **PSNR** | Overall pixel-level fidelity (higher = better) |
| **SSIM** | Perceptual/structural similarity (closer to 1 = better) |
| **Error Heatmap** | Spatial distribution of reconstruction errors |

---

## Configuration

Key constants in `model/utils.py`:

```python
IMG_SIZE   = 64    # Working resolution (all images resized to 64×64)
PATCH_SIZE = 8     # Each mask patch covers 8×8 pixels
```

---

## Expected Results

After training for 50 epochs with 50 % masking:

| Metric | Typical Range |
|--------|--------------|
| Train Loss (MSE) | 0.005 – 0.010 |
| Val Loss (MSE) | 0.008 – 0.015 |
| PSNR | 22 – 30 dB |
| SSIM | 0.75 – 0.92 |

Higher mask ratios make reconstruction harder (lower metrics).

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA GPU recommended (but CPU training works — just slower)

---

## License

This project is provided for educational and research purposes.
