import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for 3D Medical Images
    Captures long-range dependencies along spatial directions (D, H, W).
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = nn.Hardswish()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, d, h, w = x.size()
        
        x_d = self.pool_d(x)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)

        # Concatenate features for efficient processing
        # Note: Implementation details may vary based on exact tensor shapes
        y = torch.cat([x_d, x_h, x_w], dim=2) 
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_d_out, x_h_out, x_w_out = torch.split(y, [d, h, w], dim=2)

        a_d = torch.sigmoid(self.conv_d(x_d_out))
        a_h = torch.sigmoid(self.conv_h(x_h_out))
        a_w = torch.sigmoid(self.conv_w(x_w_out))

        out = identity * a_d * a_h * a_w
        return out

class LocalWindowAttention(nn.Module):
    """
    Local Window Attention (LWA)
    Focuses on fine-grained details within local 3D windows.
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # Window Partitioning logic (Simplified for readability)
        # Rearrange to (num_windows, window_size*window_size*window_size, C)
        x_windows = rearrange(x, 'b c (d wd) (h wh) (w ww) -> (b d h w) (wd wh ww) c', 
                              wd=self.window_size[0], wh=self.window_size[1], ww=self.window_size[2])
        
        qkv = self.qkv(x_windows).reshape(x_windows.shape[0], x_windows.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        x_windows = (attn @ v).transpose(1, 2).reshape(x_windows.shape[0], x_windows.shape[1], -1)
        x_windows = self.proj(x_windows)

        # Reverse Partitioning
        x = rearrange(x_windows, '(b d h w) (wd wh ww) c -> b c (d wd) (h wh) (w ww)', 
                      d=D//self.window_size[0], h=H//self.window_size[1], w=W//self.window_size[2],
                      wd=self.window_size[0], wh=self.window_size[1], ww=self.window_size[2])
        return x