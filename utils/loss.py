import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

class RCLLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(RCLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, fa, fb, fc, mask):
    
        if mask.shape[2:] != fa.shape[2:]:
            mask = F.interpolate(mask.float(), size=fa.shape[2:], mode='nearest')
        

        def get_region_vector(feat, m):

            sum_feat = torch.sum(feat * m, dim=(2, 3, 4)) # (B, C)
            
            num_pixels = torch.sum(m, dim=(2, 3, 4)) + 1e-8
            
            return sum_feat / num_pixels

        va = get_region_vector(fa, mask)
        vb = get_region_vector(fb, mask)
        
        inv_mask = 1.0 - mask
        vc = get_region_vector(fc, inv_mask) 

        va = F.normalize(va, dim=1)
        vb = F.normalize(vb, dim=1)
        vc = F.normalize(vc, dim=1)

        pos_score = torch.sum(va * vb, dim=1) / self.temperature
        neg_score = torch.sum(va * vc, dim=1) / self.temperature

        numerator = torch.exp(pos_score)
        denominator = numerator + torch.exp(neg_score) + 1e-8
        loss = -torch.log(numerator / denominator)
        
        return loss.mean()

class HybridLoss(nn.Module):
    """
    Combined Loss: Dice Loss (Segmentation) + alpha * RCL Loss
    """
    def __init__(self, alpha=0.1):
        super(HybridLoss, self).__init__()
        self.dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        self.rcl_loss = RCLLoss()
        self.alpha = alpha

    def forward(self, pred_logits, target, fa, fb, fc):
        seg_loss = self.dice_loss(pred_logits, target)
        
        con_loss = self.rcl_loss(fa, fb, fc, target) 
        
        total_loss = seg_loss + self.alpha * con_loss
        return total_loss, seg_loss, con_loss