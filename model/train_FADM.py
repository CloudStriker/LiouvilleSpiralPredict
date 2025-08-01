"""
Training script for a Frequency-Aware Diffusion Model (FADM) on the
Ulam-Liouville spiral wavelet dataset.

This script performs the following steps:
1.  Loads the wavelet-decomposed .npy files created by create_fadm_dataset.py.
2.  Initializes a UNet model from Hugging Face Diffusers, adapted for
    multi-channel wavelet input.
3.  Implements a basic diffusion training loop in the wavelet domain.
4.  Trains the model to predict the noise added to the wavelet sub-bands.
5.  Saves the fine-tuned UNet model for later use in an inference pipeline.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

# --- Dataset Class to Load Pre-computed Wavelet Data ---
class FADMDataset(Dataset):
    """
    Loads pre-computed wavelet .npy files and their corresponding masks.
    """
    def __init__(self, data_root):
        self.data_root = data_root
        self.wavelet_dir = os.path.join(data_root, "wavelets")
        self.mask_dir = os.path.join(data_root, "masks")
        
        # Assume filenames match between directories
        self.wavelet_files = sorted([f for f in os.listdir(self.wavelet_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.wavelet_files)

    def __getitem__(self, idx):
        # Load the 4-channel wavelet data
        wavelet_path = os.path.join(self.wavelet_dir, self.wavelet_files[idx])
        wavelet_data = np.load(wavelet_path)
        
        # Load the corresponding mask
        mask_filename = self.wavelet_files[idx].replace('.npy', '.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)
        mask_image = Image.open(mask_path).convert('L')
        
        # Resize mask to match the wavelet sub-band size
        # Wavelet decomposition halves the dimensions.
        wavelet_h, wavelet_w = wavelet_data.shape[1], wavelet_data.shape[2]
        mask_image_resized = mask_image.resize((wavelet_w, wavelet_h), Image.NEAREST)
        mask_resized_array = np.array(mask_image_resized, dtype=np.float32) / 255.0

        return {
            "wavelet": torch.from_numpy(wavelet_data).float(),
            "mask": torch.from_numpy(mask_resized_array).unsqueeze(0).float()
        }

def main():
    parser = argparse.ArgumentParser(description="Train a FADM UNet model.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the root of the FADM dataset (containing train/test folders).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the wavelet sub-bands (e.g., 512 crop -> 256 bands).")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for AdamW optimizer.")
    args = parser.parse_args()

    # --- 1. Setup Accelerator ---
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs")
    )

    # --- 2. Load Dataset ---
    train_dataset = FADMDataset(data_root=os.path.join(args.dataset_dir, "train"))
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # --- 3. Define Model, Scheduler, and Optimizer ---
    # The UNet input channels must match the wavelet data (4 channels: LL, LH, HL, HH)
    # We add 1 channel for the mask, so 5 channels in total.
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # --- 4. Prepare everything with Accelerator ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # --- 5. Training Loop ---
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            clean_wavelets = batch["wavelet"]
            mask = batch["mask"]
            
            noise = torch.randn(clean_wavelets.shape).to(accelerator.device)
            bs = clean_wavelets.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=accelerator.device).long()
            noisy_wavelets = noise_scheduler.add_noise(clean_wavelets, noise, timesteps)
            
            # Create the model input: noisy wavelets concatenated with the mask
            model_input = torch.cat([noisy_wavelets, mask], dim=1)

            with accelerator.accumulate(model):
                noise_pred = model(model_input, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
                accelerator.backward(loss)
                
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == args.num_epochs - 1:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
                accelerator.save_state(save_path)
                print(f"Checkpoint saved to {save_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
