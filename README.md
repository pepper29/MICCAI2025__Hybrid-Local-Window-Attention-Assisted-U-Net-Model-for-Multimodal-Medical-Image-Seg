
<div align=center>
  <h1>(MICCAI2025) Hybrid Local Window Attention Assisted U Net - Seyong Jin</h1>
  <img src=framework.png width=800>
  <p>implementation for multimodal medical image segmentation</p>
</div>

This repository provides a partial implementation of the proposed model, not the full source code of the paper.

## Paper
Full Paper Link: https://link.springer.com/chapter/10.1007/978-3-032-04937-7_22

## Introduction
This project introduces a medical image segmentation model optimized for brain tumors. The architecture uses local window attention and contrastive learning to process multimodal MRI data efficiently.

### LWSA and LWCA
LWSA handles intra modality relationships while LWCA fuses features across different modalities using window based mechanisms.

``` python
class WindowAttention3D(nn.Module):
    def __init__(self, channels, window_size=4):
        super().__init__()
        self.window_size = window_size
        self.qkv = nn.Conv3d(channels, channels * 3, 1)

    def forward(self, x):
        # Local window self attention and cross attention logic
        return x
```

### ## Detailed Modules

### 1. DRL Decoupled Representation Learning
The DRL module consists of multiple branches to extract modality specific and shared features from the bottleneck layer.

```python
import torch
import torch.nn as nn

class DisentangledRepresentationLearning(nn.Module):
    def __init__(self, in_ch=512, out_ch_list=[128, 128, 128]):
        super().__init__()
        self.branches = nn.ModuleList()
        for out_ch in out_ch_list:
            self.branches.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

    def forward(self, x):
        # Generates multiple feature maps representing different regions
        return [branch(x) for branch in self.branches]
```

### 2. RCL Region aware Contrastive Learning
This module implements the contrastive loss function to refine segmentation masks by ensuring feature consistency within the same region.
```python
import torch.nn.functional as F

def region_aware_contrastive_loss_eq7(feat_list, temperature=0.07):
    if len(feat_list) < 2:
        return torch.tensor(0.0, device=feat_list[0].device)

    def global_pool(x):
        return F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), x.size(1))

    # Calculate similarity between region features
    f0 = global_pool(feat_list[0])
    f1 = global_pool(feat_list[1])
    f2 = global_pool(feat_list[2]) if len(feat_list) > 2 else None

    pos_score = F.cosine_similarity(f0, f1, dim=-1) / temperature
    
    if f2 is not None:
        neg_score = F.cosine_similarity(f0, f2, dim=-1) / temperature
        loss = -torch.log(
            torch.exp(pos_score) / (torch.exp(pos_score) + torch.exp(neg_score) + 1e-8)
        )
    else:
        loss = -torch.log(torch.exp(pos_score) + 1e-8)
        
    return loss.mean()
```

### Author Contributions
Seyong Jin as the second author made significant contributions to this research:

* Participated in the idea planning and architectural design phases

* Implemented and integrated the RCL and DRL modules into the LWCA framework

* Served as the onsite presenter for this research at the MICCAI 2025 conference


