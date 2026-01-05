import torch
import torch.nn as nn
from .attention import CoordinateAttention, LocalWindowAttention
from .drl import DRLModule

class HybridLWAUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=32):
        super(HybridLWAUNet, self).__init__()
        
        # Encoder
        self.inc = self.conv_block(in_channels, base_filters)
        self.down1 = self.down_block(base_filters, base_filters*2)
        self.down2 = self.down_block(base_filters*2, base_filters*4)
        self.down3 = self.down_block(base_filters*4, base_filters*8)
        
        # Bottleneck
        self.bottleneck_conv = self.conv_block(base_filters*8, base_filters*16)
        
        # --- Paper Contribution: Hybrid Attention ---
        self.ca_block = CoordinateAttention(base_filters*16, base_filters*16)
        # Assuming LWA keeps dimensions same
        self.lwa_block = LocalWindowAttention(base_filters*16, window_size=(4,4,4), num_heads=8)
        
        # --- Paper Contribution: DRL Module ---
        # Takes bottleneck features and disentangles them
        self.drl = DRLModule(base_filters*16, base_filters*16)
        
        # Decoder
        # Paper suggests Decoder uses features from DRL. 
        # We will use 'fa' (anchor) as the main feature passed to decoder.
        self.up3 = self.up_block(base_filters*16, base_filters*8)
        self.up2 = self.up_block(base_filters*8, base_filters*4)
        self.up1 = self.up_block(base_filters*4, base_filters)
        
        self.outc = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool3d(2),
            self.conv_block(in_ch, out_ch)
        )

    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            self.conv_block(out_ch + out_ch, out_ch) 
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.bottleneck_conv(x4)
        x5 = self.ca_block(x5) # Coordinate Attention
        # x5 = self.lwa_block(x5) # Local Window Attention (Optional placement based on paper details)
        
        # --- DRL Disentanglement ---
        fa, fb, fc = self.drl(x5)
        
        # Decoder (Using fa as the main feature stream)
        d3 = self.up3[0](fa)
        d3 = self.up3[1](d3)
        d3 = torch.cat([d3, x4], dim=1) # Skip connection
        d3 = self.up3[2](d3)
        
        d2 = self.up2[0](d3)
        d2 = self.up2[1](d2)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.up2[2](d2)
        
        d1 = self.up1[0](d2)
        d1 = self.up1[1](d1)
        d1 = torch.cat([d1, x2], dim=1)
        d1 = self.up1[2](d1)
        
        logits = self.outc(d1)
        
        # Return logits AND DRL features for loss calculation
        return logits, fa, fb, fc