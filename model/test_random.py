"""
Inference script modified to generate a RANDOM BASELINE for comparison,
instead of using a trained FADM model.

This script performs the following steps:
1.  Loads a target image and mask.
2.  Fills the masked area with random pixels (25% white, 75% black).
3.  Saves a visual comparison with detailed quantitative analysis to
    establish a baseline performance score.
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# --- NEW: Random Fill Function (Replaces FADM) ---
def fill_mask_randomly(image, mask, white_ratio=0.75):
    """
    Fills the masked area of an image with random black and white pixels
    at a specified ratio.
    """
    # Convert images to numpy arrays
    image_array = np.array(image)
    mask_array = np.array(mask)

    # Make a copy of the original image to modify
    result_array = image_array.copy()

    # Find the coordinates of the area to be filled (where the mask is white)
    masked_coords = np.where(mask_array > 128)
    total_pixels_in_mask = len(masked_coords[0])

    if total_pixels_in_mask == 0:
        return Image.fromarray(result_array, 'RGB')

    # Calculate the number of white and black pixels for the fill
    num_white_pixels = int(total_pixels_in_mask * white_ratio)
    num_black_pixels = total_pixels_in_mask - num_white_pixels

    # Create a 1D array of pixel values to fill the mask
    fill_pixels = np.array([255] * num_white_pixels + [0] * num_black_pixels, dtype=np.uint8)
    
    # Shuffle the pixels to make the distribution random
    np.random.shuffle(fill_pixels)

    # Place the random pixels into the masked region
    # We need to do this for all 3 color channels (R, G, B)
    result_array[masked_coords[0], masked_coords[1]] = np.stack([fill_pixels, fill_pixels, fill_pixels], axis=1)

    # Convert the array back to a PIL Image
    return Image.fromarray(result_array, 'RGB')


def main():
    parser = argparse.ArgumentParser(description="Perform a baseline random inpainting test.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image from your test set.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image from your test set.")
    parser.add_argument("--output_path", type=str, default="baseline_inpainting_result.png", help="Path to save the output comparison image.")
    args = parser.parse_args()

    # --- 1. Load Data ---
    print("Loading image and mask for baseline test...")
    input_image = Image.open(args.image_path).convert("RGB")
    mask_image = Image.open(args.mask_path).convert("L")

    # --- 2. Run Random Fill (Instead of FADM Inference) ---
    print("Performing random inpainting (75% white, 25% black)...")
    result_image = fill_mask_randomly(input_image, mask_image, white_ratio=0.75)
    print("Random fill complete.")

    # --- 3. Quantitative Analysis ---
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

    # --- 4. Create Visualizations ---
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

    # --- 5. Save Final Comparison Image ---
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

    panel_titles = ["Masked Input", "Random Baseline", "Error Map", "Original Image"]
    for i, title in enumerate(panel_titles):
        text_position = (panel_width * i + panel_width // 2, title_area_height // 2)
        draw.text(text_position, title, font=title_font, fill="yellow", anchor="mm")

    text_content = (
        f"Inpainting Performance Analysis (Random Baseline)\n\n"
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
    print(f"\nDone! Baseline comparison result saved as: {args.output_path}")

if __name__ == "__main__":
    main()
