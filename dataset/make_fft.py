import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Create a Magnitude Spectrum FFT image from an input file.')
parser.add_argument('--image_path', type=str, help='The path to the input image file.')

args = parser.parse_args()

# Step 1: Load the grayscale image using the provided path
try:
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {args.image_path}")
except Exception as e:
    print(e)
    exit()

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

normalized_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
normalized_spectrum = normalized_spectrum.astype(np.uint8)

cv2.imwrite('fft.png', normalized_spectrum)

print(f"✅ 'fft.png' was successfully created from '{args.image_path}'.")