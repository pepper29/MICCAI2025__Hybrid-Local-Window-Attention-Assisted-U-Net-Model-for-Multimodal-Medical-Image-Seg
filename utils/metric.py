import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils import MetricReduction

class BraTSMetrics:
    """
    Wrapper class to calculate metrics for BraTS Tumor Regions:
    - WT (Whole Tumor)
    - TC (Tumor Core)
    - ET (Enhancing Tumor)
    """
    def __init__(self, device='cpu'):
        self.dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=False)
        self.hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction=MetricReduction.MEAN, get_not_nans=False)
        self.device = device

    def reset(self):
        self.dice_metric.reset()
        self.hd95_metric.reset()

    def __call__(self, y_pred, y):
        """
        Args:
            y_pred: (B, C, D, H, W) - Model output (Sigmoid applied or Logits)
            y: (B, C, D, H, W) - Ground Truth
        """
        # Ensure binary inputs for metrics (Threshold at 0.5)
        y_pred = (y_pred > 0.5).float()
        
        self.dice_metric(y_pred=y_pred, y=y)
        self.hd95_metric(y_pred=y_pred, y=y)

    def get_results(self):
        # Calculate mean over batch
        dice_score = self.dice_metric.aggregate().item()
        hd95_score = self.hd95_metric.aggregate().item()
        return dice_score, hd95_score

def calculate_dice(y_pred, y):
    """Standalone function for simple Dice calculation"""
    intersection = (y_pred * y).sum()
    union = y_pred.sum() + y.sum()
    return (2. * intersection + 1e-5) / (union + 1e-5)

def calculate_hd95(y_pred, y):
    """Standalone wrapper for HD95"""
    hd = HausdorffDistanceMetric(include_background=False, percentile=95)
    return hd(y_pred=y_pred, y=y)