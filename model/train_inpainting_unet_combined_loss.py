"""
Script for training a U-Net inpainting model using Focal Loss.
Accepts dataset paths, training hyperparameters, and saves the trained model weights.
"""

import argparse
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss

        return torch.mean(F_loss) if self.reduction == 'mean' else torch.sum(F_loss)

# --- Dataset ---
class InpaintingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, fft_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        self.fft_paths = sorted(glob.glob(os.path.join(fft_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        fft = Image.open(self.fft_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            fft = self.transform(fft)

        masked_image = image * (1 - mask)
        model_input = torch.cat([masked_image, fft, mask], dim=0)  # 3 channels
        target = image
        return model_input, target, mask

# --- U-Net ---
class UNetWithBN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
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

# --- Main ---
def main(args):
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    dataset = InpaintingDataset(
        image_dir=os.path.join(args.data_root, 'train/images'),
        mask_dir=os.path.join(args.data_root, 'train/masks'),
        fft_dir=os.path.join(args.data_root, 'train/ffts'),
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = UNetWithBN(in_channels=3, out_channels=1).to(device)
    focal_loss = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for model_input, target, mask in progress:
            model_input, target, mask = model_input.to(device), target.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(model_input)
            # Compute only focal loss on masked regions
            focal = focal_loss(output * mask, target * mask)
            loss = focal
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=running_loss / len(progress))

    save_path = "inpainting_unet_3ch.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# --- CLI args ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset (should contain train/images etc).")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha for Focal Loss")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma for Focal Loss")
    parser.add_argument("--lambda_f", type=float, default=1.0, help="Weight for Focal Loss")
    args = parser.parse_args()

    main(args)