import argparse
import os
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    NormalizeIntensityd, ToTensord
)
from models.network import HybridLWAUNet

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Define Transforms for Inference
    eval_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image"])
    ])

    # 2. Load Model
    model = HybridLWAUNet(in_channels=4, out_channels=3).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weights from {args.model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    model.eval()

    # 3. Load Input Image
    # Dictionary format required by MONAI transforms
    data = {"image": args.input_path}
    data = eval_transforms(data)
    image_tensor = data["image"].unsqueeze(0).to(device)  # Add batch dimension (B, C, D, H, W)

    # 4. Inference (Sliding Window)
    # ROI size should match the training patch size (e.g., 128x128x128)
    roi_size = (128, 128, 128)
    sw_batch_size = 4
    
    print("Starting inference...")
    with torch.no_grad():
        output = sliding_window_inference(
            image_tensor, roi_size, sw_batch_size, model, overlap=0.5
        )
        # Apply Sigmoid for multi-label segmentation
        output = torch.sigmoid(output)
        output = (output > 0.5).float()

    # 5. Save Prediction
    output_np = output.squeeze().cpu().numpy() # (C, D, H, W)
    
    # Load original image to get affine matrix
    original_img = nib.load(args.input_path)
    affine = original_img.affine
    
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.basename(args.input_path).replace('.nii.gz', '_pred.nii.gz')
    save_path = os.path.join(args.output_dir, filename)

    # Permute to (D, H, W, C) for NIfTI standard if needed, or keep separate
    # Here we save as 4D NIfTI
    output_nifti = nib.Nifti1Image(output_np.transpose(1, 2, 3, 0), affine)
    nib.save(output_nifti, save_path)
    
    print(f"Inference completed. Result saved at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Hybrid LWA-UNet")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input 4D NIfTI image (C, D, H, W)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save predictions")
    
    args = parser.parse_args()
    main(args)