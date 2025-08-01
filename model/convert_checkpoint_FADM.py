"""
This script converts a checkpoint saved by Accelerate into the standard
Diffusers model format that can be loaded with .from_pretrained().
"""
import argparse
import os
import torch
from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler

def main():
    parser = argparse.ArgumentParser(description="Convert an Accelerator checkpoint to a standalone Diffusers model.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the Accelerator checkpoint directory (e.g., 'finetuned_FADM/checkpoint_epoch_10').")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final, converted model.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the wavelet sub-bands the model was trained on.")
    args = parser.parse_args()

    print(f"Converting checkpoint from: {args.checkpoint_dir}")

    # --- 1. Initialize the model and scheduler with the SAME architecture as during training ---
    # Ensure this configuration matches the one in the train_fadm.py script
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=5,  # 4 wavelet bands + 1 mask channel
        out_channels=4, # Predicts the noise for each of the 4 wavelet bands
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D",
        ),
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # --- 2. Load the state from the checkpoint using Accelerator ---
    accelerator = Accelerator()
    # We need to prepare the model with the accelerator before loading the state
    model = accelerator.prepare(model)
    accelerator.load_state(args.checkpoint_dir)
    
    # Unwrap the model to get the raw model with loaded weights
    model = accelerator.unwrap_model(model)

    # --- 3. Save the model in the standard Diffusers format ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.output_dir, "unet"))
    noise_scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))

    print(f"\nConversion complete. Final model saved to: {args.output_dir}")
    print("You can now use this directory with the inference script.")

if __name__ == "__main__":
    main()
