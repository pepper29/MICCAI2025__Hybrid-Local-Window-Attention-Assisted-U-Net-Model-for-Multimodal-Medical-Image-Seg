import torch
import torch.nn as nn

class DRLModule(nn.Module):
    """
    Disentangled Representation Learning (DRL) Module
    Extracts modality-invariant and lesion-specific features.
    Outputs three feature maps: fa (anchor), fb (positive), fc (negative)
    """
    def __init__(self, in_channels, out_channels):
        super(DRLModule, self).__init__()
        
        # Branch A: Anchor representation
        self.branch_a = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Branch B: Positive sample representation
        self.branch_b = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Branch C: Negative sample representation
        self.branch_c = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        fa = self.branch_a(x)
        fb = self.branch_b(x)
        fc = self.branch_c(x)
        return fa, fb, fc