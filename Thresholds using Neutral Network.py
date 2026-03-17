# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:11:35 2026
The provided code implements a U-Net architecture optimized for segmenting biological TIFF stacks, likely for AP2 puncta detection. 
It replaces standard binary classification with an intensity-based approach using a Softplus activation layer. This code pushes out 
a predicted mask of the images. 
@author: thompson.3962
"""
import torch
import torch.nn as nn
import tifffile as tiff
import numpy as np
from tqdm import tqdm

import torchvision.transforms.functional as TF # Ensure this is at the top of your cell

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # These names MUST match what you use in forward()
        self.conv_block1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.conv_block(512, 1024)

        # Upsampling / Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_block4 = self.conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_block3 = self.conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_block2 = self.conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_block1 = self.conv_block(128, 64)

        # The "Intensity Fix" final layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Softplus()
        )

    # Helper function to create the double-convolution blocks
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        # --- ENCODER (Downsampling) ---
        c1 = self.conv_block1(x)
        p1 = self.pool1(c1)
    
        c2 = self.conv_block2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv_block3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv_block4(p3)
        p4 = self.pool4(c4)

        # --- BOTTLENECK ---
        bn = self.bottleneck(p4)

        # --- DECODER (Upsampling) ---
        # Layer 4
        u4 = self.up4(bn)
        # FIX: Use TF (torchvision) for center_crop, not F (torch.nn.functional)
        c4_cropped = TF.center_crop(c4, [u4.size(2), u4.size(3)])
        u4 = torch.cat([u4, c4_cropped], dim=1)
        d4 = self.dec_block4(u4)

        # Layer 3
        u3 = self.up3(d4)
        c3_cropped = TF.center_crop(c3, [u3.size(2), u3.size(3)])
        u3 = torch.cat([u3, c3_cropped], dim=1)
        d3 = self.dec_block3(u3)

        # Layer 2
        u2 = self.up2(d3)
        c2_cropped = TF.center_crop(c2, [u2.size(2), u2.size(3)])
        u2 = torch.cat([u2, c2_cropped], dim=1)
        d2 = self.dec_block2(u2)

        # Layer 1
        u1 = self.up1(d2)
        c1_cropped = TF.center_crop(c1, [u1.size(2), u1.size(3)])
        u1 = torch.cat([u1, c1_cropped], dim=1)
        d1 = self.dec_block1(u1)

        # --- FINAL OUTPUT ---
        return self.final_conv(d1)

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load Model (Ensure your UNet class is defined/imported)
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('best_unet_ap2.pth', map_location=device))
model.eval()

def process_tiff_stack(input_path, output_path, threshold=0.5):
    stack = tiff.imread(input_path)
    
    # If the stack is [Slices, H, W], it's already grayscale
    num_slices = stack.shape[0]
    output_masks = []

    print(f"Processing {num_slices} slices...")

    with torch.no_grad():
        for i in tqdm(range(num_slices)):
            slice_data = stack[i]
            
            # Preprocess: Normalize to 0-1 and convert to [1, 1, H, W]
            input_tensor = torch.from_numpy(slice_data).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)

            # Predict
            output = model(input_tensor)
            
            # Since you use Softplus, the output is 'intensity'. 
            # We don't need Sigmoid here.
            intensity_mask = output.squeeze().cpu().numpy()
            
            # Create binary mask based on intensity threshold
            binary_mask = (intensity_mask > threshold).astype(np.uint8) * 255
            output_masks.append(binary_mask)

    final_stack = np.stack(output_masks, axis=0)
    tiff.imwrite(output_path, final_stack, compression='zlib')
    print(f"Success! Saved to: {output_path}")

# --- Execute ---
process_tiff_stack(
    input_path="recon_1 [RENAME] - Copy-1.tif", 
    output_path="predicted_masks_volume SIM.tif"
)