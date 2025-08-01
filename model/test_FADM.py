"""
Inference script for a trained Frequency-Aware Diffusion Model (FADM)
to perform inpainting on the Ulam-Liouville spiral.

This script performs the following steps:
1.  Loads a trained UNet model saved by train_fadm.py.
2.  Loads a target image and mask.
3.  Decomposes the target image into wavelet sub-bands.
4.  Simulates the reverse diffusion (denoising) process in the wavelet domain
    to fill in the masked areas.
5.  Reconstructs the final image using an inverse wavelet transform.
6.  Saves a visual comparison with detailed quantitative analysis.
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import pywt
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF

# --- FADM Inpainting Core Function ---
@torch.no_grad()
def inpaint_with_fadm(model, scheduler, image, mask, num_inference_steps=1000, device='cuda'):
    """
    Performs the inpainting process using a trained FADM UNet.
    """
    model.eval()

    # --- 1. Prepare Data in the Wavelet Domain ---
    image_array = np.array(image.convert('L'), dtype=np.float32)
    
    coeffs = pywt.dwt2(image_array, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    def normalize(band):
        min_val, max_val = band.min(), band.max()
        if max_val - min_val < 1e-5: return np.zeros_like(band)
        return 2.0 * (band - min_val) / (max_val - min_val) - 1.0

    clean_wavelets = torch.from_numpy(
        np.stack([normalize(LL), normalize(LH), normalize(HL), normalize(HH)], axis=0)
    ).float().unsqueeze(0).to(device)

    wavelet_h, wavelet_w = LL.shape
    mask_resized = mask.resize((wavelet_w, wavelet_h), Image.NEAREST)
    mask_tensor = TF.to_tensor(mask_resized).unsqueeze(0).to(device)

    # --- 2. Reverse Diffusion Process ---
    latents = torch.randn(clean_wavelets.shape, device=device)
    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        model_input = torch.cat([latents, mask_tensor], dim=1)
        noise_pred = model(model_input, t, return_dict=False)[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        noise_t = torch.randn(clean_wavelets.shape, device=device)
        known_area_noised = scheduler.add_noise(clean_wavelets, noise_t, torch.tensor([t]*latents.shape[0], device=device))
        latents = (1.0 - mask_tensor) * known_area_noised + mask_tensor * latents

    # --- 3. Reconstruct Image from the Wavelet Domain ---
    def denormalize(band_norm, original_band):
        min_val, max_val = original_band.min(), original_band.max()
        if max_val - min_val < 1e-5: return np.zeros_like(band_norm)
        return (band_norm + 1) / 2 * (max_val - min_val) + min_val

    denoised_bands_norm = latents.cpu().squeeze(0).numpy()
    LL_recon = denormalize(denoised_bands_norm[0], LL)
    LH_recon = denormalize(denoised_bands_norm[1], LH)
    HL_recon = denormalize(denoised_bands_norm[2], HL)
    HH_recon = denormalize(denoised_bands_norm[3], HH)

    recon_coeffs = (LL_recon, (LH_recon, HL_recon, HH_recon))
    reconstructed_array = pywt.idwt2(recon_coeffs, 'haar')
    reconstructed_array = np.clip(reconstructed_array, 0, 255).astype(np.uint8)
    reconstructed_image = Image.fromarray(reconstructed_array)
    
    return reconstructed_image

def main():
    parser = argparse.ArgumentParser(description="Perform inpainting with a trained FADM model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the final converted model directory.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image from your test set.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image from your test set.")
    parser.add_argument("--output_path", type=str, default="fadm_inpainting_result.png", help="Path to save the output comparison image.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the wavelet sub-bands the model was trained on.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Load Model and Scheduler ---
    print(f"Loading model from {args.model_dir}...")
    model = UNet2DModel.from_pretrained(args.model_dir, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    model.to(device)

    # --- 2. Load Data ---
    print("Loading image and mask...")
    input_image = Image.open(args.image_path).convert("RGB")
    mask_image = Image.open(args.mask_path).convert("L")
    
    full_size = args.resolution * 2
    if input_image.size != (full_size, full_size):
        print(f"Resizing input image to {full_size}x{full_size}...")
        input_image = input_image.resize((full_size, full_size), Image.LANCZOS)
        mask_image = mask_image.resize((full_size, full_size), Image.NEAREST)

    # --- 3. Run Inference ---
    print("Performing FADM inpainting...")
    result_image = inpaint_with_fadm(model, scheduler, input_image, mask_image, device=device)
    print("Inpainting complete.")

    # --- 4. Quantitative Analysis (Copied from test_dreambooth.py) ---
    original_array = np.array(input_image.convert("L"))
    result_array = np.array(result_image.convert("L"))
    mask_array = np.array(mask_image)
    masked_coords = np.where(mask_array > 128)
    original_pixels = original_array[masked_coords]
    result_pixels = result_array[masked_coords]
    original_binary = (original_pixels > 128).astype(int)
    result_binary = (result_pixels > 128).astype(int)

    correct_pixels = np.sum(original_binary == result_binary)
    total_pixels_in_mask = len(original_binary)
    incorrect_pixels = total_pixels_in_mask - correct_pixels
    accuracy = (correct_pixels / total_pixels_in_mask) * 100 if total_pixels_in_mask > 0 else 0
    true_whites = np.sum((original_binary == 1) & (result_binary == 1))
    original_whites = np.sum(original_binary == 1)
    true_blacks = np.sum((original_binary == 0) & (result_binary == 0))
    original_blacks = np.sum(original_binary == 0)
    false_whites = original_blacks - true_blacks
    false_blacks = original_whites - true_whites

    # --- 5. Create Visualizations (Copied and adapted) ---
    error_map_array = np.full((input_image.height, input_image.width, 3), 0, dtype=np.uint8)
    CORRECT_COLOR, INCORRECT_COLOR = [255, 255, 255], [255, 0, 0]
    correct_mask = (original_binary == result_binary)
    incorrect_mask = (original_binary != result_binary)
    coords_y, coords_x = masked_coords
    error_map_array[coords_y[correct_mask], coords_x[correct_mask]] = CORRECT_COLOR
    error_map_array[coords_y[incorrect_mask], coords_x[incorrect_mask]] = INCORRECT_COLOR
    error_map_image = Image.fromarray(error_map_array, 'RGB')

    result_binary_img = result_image.convert("L").point(lambda p: 255 if p > 128 else 0, '1')
    binarized_result_image = result_binary_img.convert("RGB")
    
    masked_image = Image.new("RGB", input_image.size)
    masked_image.paste(input_image, (0, 0))
    masked_image.paste((0, 0, 0), (0, 0), mask_image)

    # --- 6. Save Final Comparison Image ---
    panel_width, panel_height = input_image.width, input_image.height
    title_area_height = 40
    total_width, total_height = panel_width * 5, panel_height + title_area_height
    
    comparison_image = Image.new("RGB", (total_width, total_height), "black")
    draw = ImageDraw.Draw(comparison_image)
    
    y_offset = title_area_height
    comparison_image.paste(masked_image, (0, y_offset))
    comparison_image.paste(binarized_result_image, (panel_width, y_offset))
    comparison_image.paste(error_map_image, (panel_width * 2, y_offset))
    comparison_image.paste(input_image, (panel_width * 3, y_offset))

    try:
        font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arialbd.ttf", 24)
    except IOError:
        print("Arial font not found. Using default font.")
        font = ImageFont.load_default()
        title_font = font

    panel_titles = ["Masked Input", "FADM Result", "Error Map", "Original Image"]
    for i, title in enumerate(panel_titles):
        text_position = (panel_width * i + panel_width // 2, title_area_height // 2)
        draw.text(text_position, title, font=title_font, fill="yellow", anchor="mm")

    text_content = (
        f"Inpainting Performance Analysis\n\n"
        f"Total Pixels in Mask: {total_pixels_in_mask:,}\n"
        f"Overall Accuracy: {accuracy:.2f}%\n\n"
        f"--- Prediction Summary ---\n"
        f"Correct Pixels: {correct_pixels:,}\n"
        f"Incorrect Pixels: {incorrect_pixels:,}\n\n"
        f"--- White Pixels Breakdown ---\n"
        f"Original: {original_whites:,}\n"
        f"  - Correct (True Whites): {true_whites:,}\n"
        f"  - Incorrect (False Blacks): {false_blacks:,}\n\n"
        f"--- Black Pixels Breakdown ---\n"
        f"Original: {original_blacks:,}\n"
        f"  - Correct (True Blacks): {true_blacks:,}\n"
        f"  - Incorrect (False Whites): {false_whites:,}"
    )
    
    text_position = (panel_width * 4 + 20, 20)
    draw.text(text_position, text_content, font=font, fill="white")

    comparison_image.save(args.output_path)
    print(f"\nDone! Comparison result saved as: {args.output_path}")
    print("Image panels (left to right): Masked -> Result -> Error Map -> Original -> Description")

if __name__ == "__main__":
    main()
