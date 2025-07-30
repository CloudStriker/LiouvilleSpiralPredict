"""
Utility script to prepare a training dataset for DreamBooth inpainting.

Given a high‑resolution source image and (optionally) its FFT, the script
randomly samples square patches from the image, applies simple
augmentations, and produces a corresponding random mask for each patch.

The resulting directory structure looks like:

    out_root/
        images/
            sample_0000.png  # RGB patch
            sample_0001.png
            ...
        masks/
            sample_0000.png  # single‑channel binary mask
            sample_0001.png
            ...
        ffts/
            sample_0000.png  # optional FFT magnitude patch (if provided)
            sample_0001.png
            ...

You can configure the number of samples to generate, the crop size,
augmentation options, and whether to include the FFT.  Masks are
generated using random rectangles or ellipses, similar to the
`random_mask` function in the HuggingFace DreamBooth inpainting example.

Run this script with Python to generate your dataset.  Example usage:

    python create_inpainting_dataset.py \
        --input-image /home/oai/share/ulam_spiral_liouville.png \
        --fft-image /home/oai/share/ulam_spiral_fft.png \
        --out-root /home/oai/share/training_data \
        --num-samples 256 \
        --crop-size 512

Note: For best results with Stable Diffusion inpainting, keep
`crop-size` equal to the model resolution (e.g. 512 or 256).
"""

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Create an inpainting dataset from a large image.")
    parser.add_argument("--input-image", type=str, required=True, help="Path to the source image (RGB).")
    parser.add_argument("--fft-image", type=str, default=None, help="Optional path to the FFT image.")
    parser.add_argument("--out-root", type=str, required=True, help="Output directory to write images, masks, and FFTs.")
    parser.add_argument("--num-samples", type=int, default=256, help="How many patches to sample.")
    parser.add_argument("--crop-size", type=int, default=512, help="Square size of each crop (e.g. 512 or 256).")
    parser.add_argument("--augment", action="store_true", help="Enable random flips and rotations.")
    parser.add_argument("--mask-ratio", type=float, default=1.0, help="Maximum ratio of mask size relative to crop size.")
    return parser.parse_args()


def random_mask(size: Tuple[int, int], ratio: float = 1.0) -> Image.Image:
    """Generate a random rectangular or elliptical mask.

    Args:
        size: (width, height) of the mask image.
        ratio: maximum size of the mask relative to the full size (between 0 and 1).

    Returns:
        A single‑channel (mode 'L') PIL Image with values 0 (background) and 255 (mask).
    """
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # Random mask dimensions up to the given ratio
    max_w = int(width * ratio)
    max_h = int(height * ratio)
    # Ensure the mask has some minimum size
    min_w = max(1, width // 8)
    min_h = max(1, height // 8)
    mask_w = random.randint(min_w, max_w)
    mask_h = random.randint(min_h, max_h)
    # Random centre
    cx = random.randint(mask_w // 2, width - mask_w // 2)
    cy = random.randint(mask_h // 2, height - mask_h // 2)
    # Random shape: rectangle or ellipse
    if random.random() < 0.5:
        # Rectangle
        x0 = cx - mask_w // 2
        y0 = cy - mask_h // 2
        x1 = cx + mask_w // 2
        y1 = cy + mask_h // 2
        draw.rectangle([x0, y0, x1, y1], fill=255)
    else:
        # Ellipse
        x0 = cx - mask_w // 2
        y0 = cy - mask_h // 2
        x1 = cx + mask_w // 2
        y1 = cy + mask_h // 2
        draw.ellipse([x0, y0, x1, y1], fill=255)
    return mask


def random_crop_with_aug(
    img: Image.Image, crop_size: int, augment: bool
) -> Tuple[Image.Image, Tuple[int, int], Tuple[bool, bool, int]]:
    """Randomly crop a square patch from the image and optionally augment it.

    Args:
        img: PIL Image to crop.
        crop_size: Size of the square crop.
        augment: If True, apply random flips/rotations.

    Returns:
        A tuple containing:
        * The cropped (and possibly augmented) PIL Image.
        * The top‑left (x, y) coordinates of the crop in the original image.
        * A tuple indicating which augmentations were applied: (flip_x, flip_y, rotations).
          rotations is an integer in {0,1,2,3}, representing 90° increments.
    """
    width, height = img.size
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Crop size {crop_size} is larger than image dimensions {img.size}."
        )
    # Random top‑left corner for the crop
    x0 = random.randint(0, width - crop_size)
    y0 = random.randint(0, height - crop_size)
    patch = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    flip_x = False
    flip_y = False
    rotations = 0
    if augment:
        # Random horizontal flip
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
            flip_x = True
        # Random vertical flip
        if random.random() < 0.5:
            patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
            flip_y = True
        # Random rotation by multiples of 90 degrees
        rotations = random.choice([0, 1, 2, 3])
        if rotations:
            patch = patch.rotate(90 * rotations, expand=True)
            # After rotation with expand=True the image may not be square; crop it back to crop_size
            patch = patch.crop((0, 0, crop_size, crop_size))
    return patch, (x0, y0), (flip_x, flip_y, rotations)


def apply_augmentations(
    img: Image.Image,
    crop_coords: Tuple[int, int],
    crop_size: int,
    augment_flags: Tuple[bool, bool, int],
) -> Image.Image:
    """Apply the same augmentations used on the RGB patch to another image (e.g. FFT).

    Args:
        img: PIL Image to apply cropping and augmentations to.
        crop_coords: (x, y) coordinates for the top‑left corner of the crop.
        crop_size: Size of the square crop.
        augment_flags: (flip_x, flip_y, rotations) as returned by random_crop_with_aug.

    Returns:
        The augmented crop from img.
    """
    x0, y0 = crop_coords
    flip_x, flip_y, rotations = augment_flags
    # Crop
    patch = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    # Apply flips
    if flip_x:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_y:
        patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
    # Apply rotations
    if rotations:
        patch = patch.rotate(90 * rotations, expand=True)
        patch = patch.crop((0, 0, crop_size, crop_size))
    return patch


def main() -> None:
    args = parse_args()
    random.seed(42)
    # Load images
    src_img = Image.open(args.input_image).convert("RGB")
    fft_img: Optional[Image.Image] = None
    if args.fft_image:
        fft_img = Image.open(args.fft_image).convert("RGB")
        if fft_img.size != src_img.size:
            raise ValueError("FFT image must have the same dimensions as the input image.")
    # Prepare output directories
    out_root = Path(args.out_root)
    images_dir = out_root / "images"
    masks_dir = out_root / "masks"
    ffts_dir = out_root / "ffts"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if fft_img is not None:
        ffts_dir.mkdir(parents=True, exist_ok=True)
    # Generate samples
    for i in range(args.num_samples):
        # Crop and augment image
        patch, coords, aug_flags = random_crop_with_aug(src_img, args.crop_size, args.augment)
        # Apply the same crop and augmentations to FFT image
        fft_patch = None
        if fft_img is not None:
            fft_patch = apply_augmentations(fft_img, coords, args.crop_size, aug_flags)
        # Generate mask for this patch
        mask = random_mask((args.crop_size, args.crop_size), ratio=args.mask_ratio)
        # Save files
        fname = f"sample_{i:04d}.png"
        patch.save(images_dir / fname)
        mask.save(masks_dir / fname)
        if fft_patch is not None:
            fft_patch.save(ffts_dir / fname)
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1} / {args.num_samples} samples")
    print(f"Dataset created at: {out_root}")


if __name__ == "__main__":
    main()
