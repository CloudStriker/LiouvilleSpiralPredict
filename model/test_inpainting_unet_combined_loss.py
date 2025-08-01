"""
Script for testing a trained U-Net with Batch Norm for inpainting.
This version accepts specific paths for image, mask, and FFT files for custom testing.
It also includes a quantitative analysis of pixel accuracy in the masked region.
"""
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import random
import matplotlib.pyplot as plt
import numpy as np

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- U-Net Architecture with Batch Normalization ---
# MUST MATCH exactly the one used during training.
class UNetWithBN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(UNetWithBN, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)

def test_model(args):
    print(f"Using device: {device}")
    
    model = UNetWithBN(in_channels=3, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model weights successfully loaded from {args.model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{args.model_path}'")
        return
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    try:
        image = Image.open(args.image_path).convert("L")
        mask = Image.open(args.mask_path).convert("L")
        fft = Image.open(args.fft_path).convert("L")
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        return

    image_tensor = transform(image)
    mask_tensor = transform(mask)
    fft_tensor = transform(fft)

    masked_image_tensor = image_tensor * (1 - mask_tensor)
    model_input = torch.cat([masked_image_tensor, fft_tensor, mask_tensor], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_tensor = model(model_input)
        prediction_tensor_sigmoid = torch.sigmoid(prediction_tensor)

    thr = args.threshold
    
# --- Quantitative Analysis ---
    original_array = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    result_array = (prediction_tensor_sigmoid.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_array = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    masked_coords = np.where(mask_array > 128)
    original_pixels = original_array[masked_coords]
    result_pixels = result_array[masked_coords]

# # Binarization using dynamic threshold
    original_binary = (original_pixels / 255.0 > thr).astype(int)
    result_binary   = (result_pixels   / 255.0 > thr).astype(int)

    correct_pixels = np.sum(original_binary == result_binary)
    total_pixels_in_mask = len(original_binary)
    incorrect_pixels = total_pixels_in_mask - correct_pixels
    accuracy = (correct_pixels / total_pixels_in_mask) * 100 if total_pixels_in_mask > 0 else 0

    true_whites = np.sum((original_binary == 1) & (result_binary == 1))
    original_whites = np.sum(original_binary == 1)
    false_blacks = original_whites - true_whites

    true_blacks = np.sum((original_binary == 0) & (result_binary == 0))
    original_blacks = np.sum(original_binary == 0)
    false_whites = original_blacks - true_blacks

# Print analysis results to console
    print("\n--- Inpainting Performance Analysis ---")
    print(f"Total Pixels in Mask: {total_pixels_in_mask:,}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Correct Pixels: {correct_pixels:,}")
    print(f"Incorrect Pixels: {incorrect_pixels:,}")
    print("\n--- White Pixels Breakdown ---")
    print(f"Original: {original_whites:,}")
    print(f"  - Correct (True Whites): {true_whites:,}")
    print(f"  - Incorrect (False Blacks): {false_blacks:,}")
    print("\n--- Black Pixels Breakdown ---")
    print(f"Original: {original_blacks:,}")
    print(f"  - Correct (True Blacks): {true_blacks:,}")
    print(f"  - Incorrect (False Whites): {false_whites:,}")
    print("-------------------------------------\n")
    
# --- Visualization ---
    to_pil = transforms.ToPILImage()
    input_vis = to_pil(masked_image_tensor.cpu())
    ground_truth_vis = to_pil(image_tensor.cpu())
    
    # Binarization of the final result for clearer visualization
    # Use dynamic threshold for BW visualization
    cutoff = thr * 255
    # Combine prediction result with original image using the mask
    final_result_tensor = image_tensor * (1 - mask_tensor) + prediction_tensor_sigmoid.squeeze(0).cpu() * mask_tensor
    # Convert to grayscale PIL for visualization
    final_result_vis_gray = to_pil(final_result_tensor)
    # Binarize with dynamic threshold for BW visualization
    final_result_vis_bw = final_result_vis_gray.point(lambda p: 255 if p > cutoff else 0, '1')

    # Creating a new canvas for 5 panels (4 images + 1 text)
    panel_width, panel_height = args.img_size, args.img_size
    title_area_height = 40
    total_width = panel_width * 5
    total_height = panel_height + title_area_height

    comparison_image = Image.new("RGB", (total_width, total_height), "black")
    draw = ImageDraw.Draw(comparison_image)
    
    y_offset = title_area_height
    # [MODIFICATION] Create masked input from PIL Image to avoid gray appearance
    masked_input_pil = image.copy().convert("RGB")
    draw_mask = ImageDraw.Draw(masked_input_pil)
    mask_pil = mask.copy()
    # Invert mask for pasting
    inverted_mask_pil = Image.eval(mask_pil, lambda x: 255 - x)
    masked_input_pil.paste((0,0,0), (0,0), mask_pil)


    comparison_image.paste(masked_input_pil, (0, y_offset))
    comparison_image.paste(final_result_vis_bw.convert("RGB"), (panel_width, y_offset))
    
    # Creating Error Map (Red for incorrect, White for correct)
    error_map_array = np.full((panel_height, panel_width, 3), 0, dtype=np.uint8) # Black background
    correct_mask = (original_binary == result_binary)
    incorrect_mask = (original_binary != result_binary)
    coords_y, coords_x = masked_coords
    error_map_array[coords_y[correct_mask], coords_x[correct_mask]] = [255, 255, 255] # White
    error_map_array[coords_y[incorrect_mask], coords_x[incorrect_mask]] = [255, 0, 0] # Red
    error_map_image = Image.fromarray(error_map_array, 'RGB')
    comparison_image.paste(error_map_image, (panel_width * 2, y_offset))

    comparison_image.paste(ground_truth_vis.convert("RGB"), (panel_width * 3, y_offset))

    # Adding text and titles
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arialbd.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        title_font = font

    panel_titles = ["Masked Input", "Inpainting Result", "Error Map", "Original Image"]
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
    draw.text((panel_width * 4 + 20, y_offset + 20), text_content, font=font, fill="white", spacing=3)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.output_dir, f"test_result_analyzed_{base_name}.png")
    comparison_image.save(save_path)
    
    print(f"\nTesting complete. Comparison image saved at: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained U-Net with quantitative analysis.")
    
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model file (e.g., './inpainting_unet_bn.pt').")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file (the ground truth).")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask file.")
    parser.add_argument("--fft_path", type=str, required=True, help="Path to the corresponding FFT image file.")
    parser.add_argument("--output_dir", type=str, default="test_results_analyzed", help="Directory to save the output test image.")
    parser.add_argument("--img_size", type=int, default=512, help="Image size used during training. Must match the training configuration.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Sigmoid threshold for binarization (0–1 range)")

    args = parser.parse_args()
    test_model(args)
