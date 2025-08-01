import numpy as np
from PIL import Image

def create_random_baseline_image(width, height, white_ratio=0.25, output_filename="baseline_image_white.png"):
    """
    Creates and saves a random black and white image with a specific ratio.
    """
    print(f"Creating a {width}x{height} image with {white_ratio*100}% white pixels...")

    # Calculate total pixels and number of white pixels
    total_pixels = width * height
    num_white_pixels = int(total_pixels * white_ratio)
    num_black_pixels = total_pixels - num_white_pixels

    # Create a 1D array with the correct number of black (0) and white (255) pixels
    pixels = np.array([0] * num_black_pixels + [255] * num_white_pixels, dtype=np.uint8)
    
    # Shuffle the array to make the pixel distribution random
    np.random.shuffle(pixels)
    
    # Reshape the 1D array back into a 2D image
    image_array = pixels.reshape((height, width))
    
    # Convert the array to a PIL Image and save it
    image = Image.fromarray(image_array, mode='L') # 'L' for grayscale
    image.save(output_filename)
    
    print(f"Baseline image saved as '{output_filename}'")

if __name__ == "__main__":
    # Define the size to match your dataset samples
    IMAGE_SIZE = 512
    
    # Define the ratio based on your analysis (approx. 25% white, 75% black)
    WHITE_PIXEL_RATIO = 0.25

    create_random_baseline_image(IMAGE_SIZE, IMAGE_SIZE, WHITE_PIXEL_RATIO)
