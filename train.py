import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import OctFoundationModelMAE, OctSegModel
from src.dataset import MockOCTDataset, OCTBScanDataset
import os
from tqdm import tqdm
import time

def get_args():
    parser = argparse.ArgumentParser(description="Train OCT Foundation Model")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "segmentation"], help="Training mode")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to B-scan images")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--signal_len", type=int, default=1024, help="Length of A-scan")
    parser.add_argument("--use_mock", action="store_true", help="Use synthetic data")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio for MAE")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to pretrained checkpoint for fine-tuning")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes for segmentation (0=background)")
    
    return parser.parse_args()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    if args.use_mock:
        print("Using Mock Dataset...")
        dataset = MockOCTDataset(num_samples=10000, signal_len=args.signal_len)
    else:
        print(f"Loading data from {args.data_path}...")
        dataset = OCTBScanDataset(args.data_path, signal_len=args.signal_len)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # Model Setup
    base_model = OctFoundationModelMAE(signal_len=args.signal_len, 
                                       embed_dim=192, depth=12, num_heads=3, 
                                       decoder_embed_dim=128, decoder_depth=4)
    
    if args.pretrained_ckpt:
        print(f"Loading pretrained weights from {args.pretrained_ckpt}")
        checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
        # Load state dict (handle potential 'model_state_dict' key)
        sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        base_model.load_state_dict(sd, strict=False)

    if args.mode == "pretrain":
        model = base_model
    else:
        # Segmentation
        model = OctSegModel(base_model, num_classes=args.num_classes)
        
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    if args.mode == "segmentation":
        criterion = nn.CrossEntropyLoss()
    
    print(f"Starting {args.mode}...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for signals, masks in pbar:
            signals = signals.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                if args.mode == "pretrain":
                    loss, _, _ = model(signals, mask_ratio=args.mask_ratio)
                else:
                    # Segmentation
                    logits = model(signals) # [B, C, L]
                    loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"{args.mode}_checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
    
    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes.")
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
