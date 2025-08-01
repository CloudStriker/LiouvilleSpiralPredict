"""
Utility script to prepare a frequency-aware (FADM-style) training dataset.

Given a high-resolution source image, this script randomly samples square
patches, applies a 2D Wavelet Transform to decompose each patch into
frequency sub-bands (LL, LH, HL, HH), and saves them as multi-channel
NumPy arrays (.npy files). It also produces a corresponding random mask.

The resulting directory structure is split into train and test sets:

    out_root/
        train/
            wavelets/       # 4-channel .npy files (LL, LH, HL, HH)
            masks/          # single-channel .png binary masks
        test/
            wavelets/
            masks/
"""

import argparse
import random
from pathlib import Path
import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import pywt
from PIL import Image, ImageDraw

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a frequency-aware inpainting dataset.")
    parser.add_argument("--input-image", type=str, required=True, help="Path to the source image.")
    parser.add_argument("--out-root", type=str, required=True, help="Output directory for the dataset.")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of patches to sample.")
    parser.add_argument("--crop-size", type=int, default=512, help="Square size of each crop.")
    parser.add_argument("--augment", action="store_true", help="Enable random flips and rotations.")
    parser.add_argument("--mask-ratio", type=float, default=0.5, help="Maximum ratio of mask size relative to crop size.")
    parser.add_argument("--test-split-ratio", type=float, default=0.3, help="Ratio of samples for the test set.")
    return parser.parse_args()

# --- NEW: WAVELET DECOMPOSITION FUNCTION ---
def image_to_wavelet_bands(img: Image.Image) -> np.ndarray:
    """
    Converts a PIL image to a 4-channel NumPy array of its wavelet sub-bands.
    """
    # Convert image to grayscale numpy array, normalized to [0, 1]
    img_array = np.array(img.convert('L'), dtype=np.float32) / 255.0
    
    # Perform a 2D Discrete Wavelet Transform
    coeffs = pywt.dwt2(img_array, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # Normalize each band to the range [-1, 1] for the model
    def normalize(band):
        min_val, max_val = band.min(), band.max()
        if max_val - min_val < 1e-5: # Avoid division by zero
            return np.zeros_like(band)
        return 2.0 * (band - min_val) / (max_val - min_val) - 1.0
        
    # Stack the normalized bands into a single 4-channel array
    return np.stack([normalize(LL), normalize(LH), normalize(HL), normalize(HH)], axis=0)

# --- Functions for cropping, masking, etc. (mostly unchanged) ---
def random_mask(size: tuple, ratio: float = 1.0) -> Image.Image:
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    max_w, max_h = int(width * ratio), int(height * ratio)
    min_w, min_h = max(1, width // 8), max(1, height // 8)
    mask_w, mask_h = random.randint(min_w, max_w), random.randint(min_h, max_h)
    cx, cy = random.randint(mask_w // 2, width - mask_w // 2), random.randint(mask_h // 2, height - mask_h // 2)
    
    if random.random() < 0.5:
        x0, y0, x1, y1 = cx - mask_w // 2, cy - mask_h // 2, cx + mask_w // 2, cy + mask_h // 2
        draw.rectangle([x0, y0, x1, y1], fill=255)
    else:
        x0, y0, x1, y1 = cx - mask_w // 2, cy - mask_h // 2, cx + mask_w // 2, cy + mask_h // 2
        draw.ellipse([x0, y0, x1, y1], fill=255)
    return mask

def random_crop_with_aug(img: Image.Image, crop_size: int, augment: bool):
    width, height = img.size
    if width < crop_size or height < crop_size:
        raise ValueError(f"Crop size {crop_size} is larger than image dimensions {img.size}.")
    x0, y0 = random.randint(0, width - crop_size), random.randint(0, height - crop_size)
    patch = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    if augment:
        if random.random() < 0.5: patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5: patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
        rotations = random.choice([0, 1, 2, 3])
        if rotations:
            patch = patch.rotate(90 * rotations, expand=True)
            patch = patch.crop((0, 0, crop_size, crop_size))
    return patch

def main() -> None:
    args = parse_args()
    random.seed(42)
    
    print("Loading source image...")
    src_img = Image.open(args.input_image).convert("RGB")

    # Prepare temporary output directories
    out_root = Path(args.out_root)
    wavelets_dir = out_root / "wavelets" # Changed from 'images'
    masks_dir = out_root / "masks"
    wavelets_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} wavelet-decomposed samples...")
    for i in range(args.num_samples):
        # Crop and augment the image patch
        patch_img = random_crop_with_aug(src_img, args.crop_size, args.augment)
        
        # --- MODIFIED: Convert patch to wavelet bands and save as .npy ---
        wavelet_bands = image_to_wavelet_bands(patch_img)
        
        # Generate a corresponding mask
        mask = random_mask((args.crop_size, args.crop_size), ratio=args.mask_ratio)
        
        # Save files
        fname_base = f"sample_{i:04d}"
        np.save(wavelets_dir / f"{fname_base}.npy", wavelet_bands)
        mask.save(masks_dir / f"{fname_base}.png")

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1} / {args.num_samples} samples")
    
    print(f"\nInitial wavelet dataset created at: {out_root}")

    # --- Split the generated data into train and test sets ---
    print("\nSplitting dataset into train and test sets...")
    
    train_dir = out_root / "train"
    test_dir = out_root / "test"

    (train_dir / "wavelets").mkdir(parents=True, exist_ok=True)
    (train_dir / "masks").mkdir(parents=True, exist_ok=True)
    (test_dir / "wavelets").mkdir(parents=True, exist_ok=True)
    (test_dir / "masks").mkdir(parents=True, exist_ok=True)

    all_files_base = sorted([f.stem for f in wavelets_dir.glob('*.npy')])
    train_files, test_files = train_test_split(all_files_base, test_size=args.test_split_ratio, random_state=42)

    def move_files(file_list, destination_dir):
        for fname_base in file_list:
            shutil.move(wavelets_dir / f"{fname_base}.npy", destination_dir / "wavelets" / f"{fname_base}.npy")
            shutil.move(masks_dir / f"{fname_base}.png", destination_dir / "masks" / f"{fname_base}.png")

    move_files(train_files, train_dir)
    move_files(test_files, test_dir)
    
    os.rmdir(wavelets_dir)
    os.rmdir(masks_dir)

    print(f"Split complete. Training samples: {len(train_files)}, Testing samples: {len(test_files)}")
    print(f"FADM-style dataset is now ready in: {out_root}")


if __name__ == "__main__":
    main()
