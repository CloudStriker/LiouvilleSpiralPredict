"""
Script to count the number of 'white' and 'black' pixels in an image.

This script loads an image, converts it to grayscale, and then uses a
threshold to classify each pixel as either black or white. It then
prints the total count for each category.
"""

import argparse
import numpy as np
from PIL import Image

def count_image_pixels(image_path, threshold=128):
    """
    Loads an image, counts its black and white pixels based on a threshold,
    and returns the counts.

    Args:
        image_path (str): The path to the input image file.
        threshold (int): The grayscale value (0-255) to distinguish black
                         from white. Pixels with values > threshold are
                         considered white.

    Returns:
        dict: A dictionary containing the counts for 'white_pixels',
              'black_pixels', and 'total_pixels', or None if the
              image cannot be opened.
    """
    try:
        # Open the image and convert it to grayscale ('L' mode)
        img = Image.open(image_path).convert('L')
        
        # Convert the image to a NumPy array for efficient processing
        img_array = np.array(img)
        
        # Get the total number of pixels
        total_pixels = img_array.size
        
        # Count white pixels (pixels with a value greater than the threshold)
        white_pixels = np.sum(img_array > threshold)
        
        # Black pixels are the remainder
        black_pixels = total_pixels - white_pixels
        
        return {
            'white_pixels': white_pixels,
            'black_pixels': black_pixels,
            'total_pixels': total_pixels
        }

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    """
    Main function to parse command-line arguments and run the pixel counter.
    """
    parser = argparse.ArgumentParser(description="Count white and black pixels in an image.")
    
    parser.add_argument("-i", "--image_path", type=str, required=True,
                        help="Path to the input image file.")
                        
    parser.add_argument("-t", "--threshold", type=int, default=128,
                        help="Grayscale threshold (0-255) to define white vs. black. Default is 128.")

    args = parser.parse_args()

    print(f"Processing image: {args.image_path}")
    print(f"Using threshold: {args.threshold}\n")

    pixel_counts = count_image_pixels(args.image_path, args.threshold)
    
    if pixel_counts:
        total = pixel_counts['total_pixels']
        white = pixel_counts['white_pixels']
        black = pixel_counts['black_pixels']
        
        # Calculate percentages
        white_percent = (white / total) * 100 if total > 0 else 0
        black_percent = (black / total) * 100 if total > 0 else 0

        print("--- Pixel Count Results ---")
        print(f"Total Pixels: {total:,}")
        print(f"White Pixels (> {args.threshold}): {white:,} ({white_percent:.2f}%)")
        print(f"Black Pixels (<= {args.threshold}): {black:,} ({black_percent:.2f}%)")
        print("---------------------------")

if __name__ == "__main__":
    main()
