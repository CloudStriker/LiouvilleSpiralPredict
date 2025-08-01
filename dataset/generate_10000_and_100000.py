import numpy as np
from PIL import Image
import math
import os
import numba
from functools import lru_cache
import multiprocessing as mp

# Use numba to dramatically speed up the prime factorization
@numba.jit(nopython=True, cache=True)
def prime_factors_count_numba(n: int) -> int:
    """Calculate the number of prime factors of n, with multiplicity."""
    if n <= 1:
        return 0
    if n == 2 or n == 3:
        return 1
        
    count = 0
    # Check if divisible by 2
    while n % 2 == 0:
        n //= 2
        count += 1
        
    # Only check odd numbers
    i = 3
    while i * i <= n:
        while n % i == 0:
            n //= i
            count += 1
        i += 2  # Only check odd numbers
        
    if n > 1:
        count += 1
    return count

# Precompute small values in a lookup table for even faster access
LIOUVILLE_CACHE_SIZE = 1000000
liouville_lookup = np.zeros(LIOUVILLE_CACHE_SIZE, dtype=np.int8)

def initialize_lookup():
    """Precompute Liouville values for small numbers"""
    for i in range(LIOUVILLE_CACHE_SIZE):
        liouville_lookup[i] = 1 if prime_factors_count_numba(i) % 2 == 0 else -1
    print(f"Precomputed {LIOUVILLE_CACHE_SIZE} Liouville values")

# For larger values, use numba-accelerated function
@numba.jit(nopython=True, cache=True)
def liouville_numba(n: int) -> int:
    """Calculate Liouville function value: λ(n) = (-1)^Ω(n)."""
    return 1 if prime_factors_count_numba(n) % 2 == 0 else -1

def liouville(n: int) -> int:
    """Compute Liouville function with lookup table for small values."""
    if n < LIOUVILLE_CACHE_SIZE:
        return liouville_lookup[n]
    return liouville_numba(n)

@numba.jit(nopython=True, parallel=True)
def process_chunk_inner(height, width, x_start, y_start, y_offset):
    """Numba-accelerated inner loop for processing a chunk"""
    chunk = np.zeros((height, width), dtype=np.uint8)
    
    for i in numba.prange(height):
        for j in range(width):
            global_x = x_start + j
            global_y = y_start + i + y_offset
            n = global_y * 1000000 + global_x
            
            # Direct calculation for numba
            if n < LIOUVILLE_CACHE_SIZE:
                val = liouville_lookup[n]
            else:
                val = 1 if prime_factors_count_numba(n) % 2 == 0 else -1
                
            chunk[i, j] = 255 if val == 1 else 0
            
    return chunk

def process_chunk(args):
    """Process a chunk of the image"""
    y_range, x_range, x_start, y_start = args
    height = len(y_range)
    width = len(x_range)
    y_offset = y_range[0]
    
    # Use numba-accelerated function for the heavy computation
    chunk = process_chunk_inner(height, width, x_start, y_start, y_offset)
    return chunk, y_range[0], 0

def generate_liouville_patch(center_x: int, center_y: int, size: int, output_path: str, num_processes=None):
    """
    Generate a square patch of the Liouville function mapped to an image.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    half_size = size // 2
    x_start = center_x - half_size
    y_start = center_y - half_size

    # Create numpy array to store pixel values
    img_array = np.zeros((size, size), dtype=np.uint8)

    print(f"Generating Liouville function patch around ({center_x},{center_y})...")
    print(f"Using {num_processes} parallel processes...")

    # Split the work into chunks for parallel processing
    chunk_size = size // num_processes
    chunks = []
    
    for i in range(0, size, chunk_size):
        y_end = min(i + chunk_size, size)
        chunks.append(
            (range(i, y_end), range(0, size), x_start, y_start)
        )
    
    # Process chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Assemble the results
    for chunk, y_offset, x_offset in results:
        img_array[y_offset:y_offset+chunk.shape[0], x_offset:x_offset+chunk.shape[1]] = chunk

    # Save as image
    img = Image.fromarray(img_array, mode='L')  # Mode 'L' for grayscale
    img.save(output_path)
    print(f"Image saved at: {output_path}")

if __name__ == "__main__":
    # Initialize the lookup table first
    initialize_lookup()
    
    # Use 80% of available CPUs
    num_cpus = max(1, int(mp.cpu_count() * 0.8))
    generate_liouville_patch(10000, 10000, 5000, "liouville_patch_10000_10000.png", num_cpus)
    generate_liouville_patch(100000, 100000, 5000, "liouville_patch_100000_100000.png", num_cpus)
    # generate_liouville_patch(1000, 1000, 512, "test_patch.png")


