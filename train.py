import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import BraTSDataset, get_transforms
from models.network import HybridLWAUNet
from utils.loss import HybridLoss
from tqdm import tqdm

"""
[MICCAI 2025] Hybrid Local-Window-Attention-Assisted U-Net
: This is a partial implementation, not the full source code of the paper.

This script serves as a prototype to verify the training logic and architecture 
of the proposed model. It includes the training loop with Hybrid Loss (Segmentation + RCL).
"""

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    train_ds = BraTSDataset(args.json_path, phase='training', transform=get_transforms('train'))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    model = HybridLWAUNet(in_channels=4, out_channels=3).to(device)
    
    # Optimizer & Hybrid Loss (Seyong Jin's Implementation)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = HybridLoss(alpha=0.1).to(device) # alpha weighs the contrastive loss
    
    print(f"Start training Hybrid LWA-UNet (w/ RCL & DRL)")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_seg_loss = 0
        epoch_con_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass returns logits AND features for DRL/RCL
                logits, fa, fb, fc = model(inputs)
                
                # Calculate Hybrid Loss
                loss, seg_l, con_l = criterion(logits, labels, fa, fb, fc)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_seg_loss += seg_l.item()
                epoch_con_loss += con_l.item()
                
                pbar.set_postfix({'Total': loss.item(), 'Seg': seg_l.item(), 'RCL': con_l.item()})
        
        print(f"Epoch {epoch+1} Results - Mean Loss: {epoch_loss/len(train_loader):.4f} "
              f"(Seg: {epoch_seg_loss/len(train_loader):.4f}, RCL: {epoch_con_loss/len(train_loader):.4f})")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="./data/dataset.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)