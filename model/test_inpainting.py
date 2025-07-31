import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- SETUP ---
# 1. Path to your fine-tuned model please use the path where your model is saved
# if you run the training script, it will save the model in the finetuned_model directory.
# example: model_path = "/home/user/liouville/finetuned_model"
# or you can use from Huggingface ClickNoow/LiouvilleSpiralPredict
# example: model_path = "ClickNoow/LiouvilleSpiralPredict"
model_path = "ClickNoow/LiouvilleSpiralPredict"

# 2. The prompt you used during training
prompt = "a photo of ulamsprial pattern"

# 3. Prepare an input image and mask for testing
#    We will use one of the samples from your dataset
#    Change these paths if necessary
#    Please use full paths to the images and masks
image_path = "/home/user/liouville/dataset/ulam_training_data/test/images/sample_0006.png"
mask_path = "/home/user/liouville/dataset/ulam_training_data/test/masks/sample_0006.png"

# --- INFERENCE PROCESS ---
print(f"Loading model from: {model_path}")
# Load the fine-tuned pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

print("Loading input image and mask...")
# Load the original image and its corresponding mask
input_image = Image.open(image_path).convert("RGB")
input_mask = Image.open(mask_path).convert("L")

# Create a masked image for visualization (the area to be filled is black)
masked_image = Image.new("RGB", input_image.size)
masked_image.paste(input_image, (0, 0))
masked_image.paste((0, 0, 0), (0, 0), input_mask)

print("Running the inpainting process...")
# Run the inpainting pipeline
result_image = pipe(
    prompt=prompt,
    image=input_image,
    mask_image=input_mask,
    num_inference_steps=50
).images[0]

# --- QUANTITATIVE ANALYSIS ---
# Convert images to numpy arrays for analysis
original_array = np.array(input_image.convert("L")) # Convert original to grayscale for comparison
result_array = np.array(result_image.convert("L"))
mask_array = np.array(input_mask)

# Find the coordinates of the masked (inpainted) area
masked_coords = np.where(mask_array > 128) 

# Extract the pixel values from the original and result images in the masked area
original_pixels = original_array[masked_coords]
result_pixels = result_array[masked_coords]

# Threshold pixels to be either black (0) or white (1)
original_binary = (original_pixels > 128).astype(int)
result_binary = (result_pixels > 128).astype(int)

# Calculate metrics
correct_pixels = np.sum(original_binary == result_binary)
total_pixels_in_mask = len(original_binary)
incorrect_pixels = total_pixels_in_mask - correct_pixels # Calculate incorrect pixels
accuracy = (correct_pixels / total_pixels_in_mask) * 100 if total_pixels_in_mask > 0 else 0

true_whites = np.sum((original_binary == 1) & (result_binary == 1))
original_whites = np.sum(original_binary == 1)

true_blacks = np.sum((original_binary == 0) & (result_binary == 0))
original_blacks = np.sum(original_binary == 0)

false_whites = original_blacks - true_blacks # Originally black, predicted white
false_blacks = original_whites - true_whites # Originally white, predicted black

# --- NEW: VISUAL ERROR MAP CREATION (Simplified Black/White/Red) ---
# Create a base RGB array for the error map
error_map_array = np.full((input_image.height, input_image.width, 3), 0, dtype=np.uint8) # Black background

# Define colors for visualization
CORRECT_COLOR = [255, 255, 255]  # White
INCORRECT_COLOR = [255, 0, 0]    # Red

# Create boolean masks for correct and incorrect predictions
correct_mask = (original_binary == result_binary)
incorrect_mask = (original_binary != result_binary)

# Apply colors to the error map using efficient numpy indexing
coords_y, coords_x = masked_coords
error_map_array[coords_y[correct_mask], coords_x[correct_mask]] = CORRECT_COLOR
error_map_array[coords_y[incorrect_mask], coords_x[incorrect_mask]] = INCORRECT_COLOR

# Convert the numpy array back to a PIL Image
error_map_image = Image.fromarray(error_map_array, 'RGB')

# --- NEW: Binarize the result image for clear black and white display ---
result_gray = result_image.convert("L")
# Create a pure black and white image using a threshold
result_binary_img = result_gray.point(lambda p: 255 if p > 128 else 0, '1')
binarized_result_image = result_binary_img.convert("RGB")


# --- SAVING THE RESULT (UPDATED FOR LAYOUT WITH TITLES ON TOP) ---
# Define dimensions and create a new canvas with space for titles
panel_width = input_image.width
panel_height = input_image.height
title_area_height = 40 # Extra space at the top for titles
total_width = panel_width * 5 # 4 panels for images, 1 for text
total_height = panel_height + title_area_height

comparison_image = Image.new("RGB", (total_width, total_height), "black")
draw = ImageDraw.Draw(comparison_image)

# Paste the 4 image panels with an offset from the top
y_offset = title_area_height
comparison_image.paste(masked_image, (0, y_offset))
comparison_image.paste(binarized_result_image, (panel_width, y_offset))
comparison_image.paste(error_map_image, (panel_width * 2, y_offset))
comparison_image.paste(input_image, (panel_width * 3, y_offset))

# --- ADD TITLES AND DESCRIPTION TEXT TO THE IMAGE ---
try:
    # Use a common truetype font if available
    font = ImageFont.truetype("arial.ttf", 18)
    title_font = ImageFont.truetype("arialbd.ttf", 24) # Bold font for titles
except IOError:
    # Fallback to a default bitmap font if the specific font is not found
    print("Arial font not found. Using default font.")
    font = ImageFont.load_default()
    title_font = font

# --- MODIFIED: Add titles to the top area ---
panel_titles = ["Masked Input", "Inpainting Result", "Error Map", "Original Image"]
for i, title in enumerate(panel_titles):
    # Center the title in its respective panel area
    text_position = (panel_width * i + panel_width // 2, title_area_height // 2)
    draw.text(text_position, title, font=title_font, fill="yellow", anchor="mm")

# --- MODIFIED: Detailed text content with summary ---
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

# Position and draw the text in the 5th panel
text_position = (panel_width * 4 + 20, 20) # Start in the 5th panel
draw.text(text_position, text_content, font=font, fill="white")

output_path = "inpainting_result_comparison.png"
comparison_image.save(output_path)

print(f"\nDone! Comparison result saved as: {output_path}")
print("Image panels (left to right): Masked -> Result -> Error Map -> Original -> Description")
