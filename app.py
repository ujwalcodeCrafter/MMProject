"""
Flask Web Application — Masked Image Reconstruction.

Endpoints:
    GET  /              → Main page (upload + results UI)
    POST /reconstruct   → Process uploaded image, return JSON with paths & metrics
"""

import os
import uuid

import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for
from torchvision import transforms

from model.model import MaskedAutoencoder
from model.utils import (
    IMG_SIZE,
    create_patch_mask,
    apply_mask,
    calculate_psnr,
    calculate_ssim,
    generate_error_heatmap,
    tensor_to_numpy,
    save_tensor_as_image,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

# ─── Device & model ─────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: MaskedAutoencoder | None = None

# Image preprocessing (resize to model's expected input)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_model(model_path: str = 'checkpoints/best_model.pth'):
    """Load trained weights (or fall back to an untrained model)."""
    global model
    model = MaskedAutoencoder(in_channels=3).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        print(f"[INFO] Model loaded from {model_path}  "
              f"(epoch {epoch}, val_loss {val_loss})")
    else:
        print(f"[WARNING] No checkpoint at {model_path} — using untrained model.")

    model.eval()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_mask_visualization(image_tensor, mask, save_path):
    """
    Overlay the mask on the original image.
    Visible regions stay normal; masked regions get a translucent red tint.
    """
    img_np = tensor_to_numpy(image_tensor)           # (H, W, 3)
    mask_np = mask.squeeze(0).numpy()                 # (H, W)
    mask_3d = np.stack([mask_np] * 3, axis=2)         # (H, W, 3)

    overlay = img_np.copy()
    # Darken + red tint on masked regions
    overlay[mask_3d == 0] *= 0.3
    red = np.zeros_like(overlay)
    red[:, :, 0] = 0.7
    overlay = np.where(mask_3d == 0, np.clip(overlay + red * 0.5, 0, 1), overlay)

    plt.imsave(save_path, overlay.astype(np.float32))


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/reconstruct', methods=['POST'])
def reconstruct():
    """
    Accept an uploaded image + mask ratio, run reconstruction,
    and return a JSON response with image paths and quality metrics.
    """
    # ── Validate input ───────────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'error': 'No image file received.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    mask_ratio = float(request.form.get('mask_ratio', 0.5))
    mask_ratio = max(0.1, min(0.9, mask_ratio))

    try:
        uid = uuid.uuid4().hex[:8]

        # ── Save & load uploaded image ───────────────────────────────────
        upload_path = os.path.join('static', 'uploads', f'{uid}_upload.png')
        file.save(upload_path)
        image = Image.open(upload_path).convert('RGB')
        image_tensor = preprocess(image)                 # (3, H, W)

        # ── Create mask & apply ──────────────────────────────────────────
        mask, _ = create_patch_mask(mask_ratio=mask_ratio)
        masked_image = apply_mask(image_tensor, mask)

        # ── Run model inference ──────────────────────────────────────────
        with torch.no_grad():
            inp = masked_image.unsqueeze(0).to(device)   # (1, 3, H, W)
            out = model(inp)
            reconstructed = out.squeeze(0).cpu()          # (3, H, W)

        # ── Save result images ───────────────────────────────────────────
        def out_path(tag):
            return os.path.join('static', 'outputs', f'{uid}_{tag}.png')

        save_tensor_as_image(image_tensor,   out_path('original'))
        save_tensor_as_image(masked_image,   out_path('masked'))
        save_tensor_as_image(reconstructed,  out_path('reconstructed'))

        # ── Quality metrics ──────────────────────────────────────────────
        orig_np  = tensor_to_numpy(image_tensor)
        recon_np = tensor_to_numpy(reconstructed)

        psnr_val = calculate_psnr(orig_np, recon_np)
        ssim_val = calculate_ssim(orig_np, recon_np)

        # ── Heatmap & mask visualisation ─────────────────────────────────
        generate_error_heatmap(orig_np, recon_np, out_path('heatmap'))
        _save_mask_visualization(image_tensor, mask, out_path('maskvis'))

        # ── Respond ──────────────────────────────────────────────────────
        def static_url(tag):
            return url_for('static', filename=f'outputs/{uid}_{tag}.png')

        return jsonify({
            'success':            True,
            'original':           static_url('original'),
            'masked':             static_url('masked'),
            'reconstructed':      static_url('reconstructed'),
            'heatmap':            static_url('heatmap'),
            'mask_visualization': static_url('maskvis'),
            'psnr':               float(round(psnr_val, 2)),
            'ssim':               float(round(ssim_val, 4)),
            'mask_ratio':         float(mask_ratio),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print(f"[INFO] Device: {device}")
    print("[INFO] Starting Flask on http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
