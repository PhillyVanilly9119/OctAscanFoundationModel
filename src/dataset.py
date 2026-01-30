import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from glob import glob
import random

class MockOCTDataset(Dataset):
    def __init__(self, num_samples=10000, signal_len=1024):
        self.num_samples = num_samples
        self.signal_len = signal_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic A-scan: simplified multi-gaussian peaks
        x = np.linspace(0, 1, self.signal_len)
        signal = np.zeros_like(x)
        mask = np.zeros_like(x, dtype=np.int64) # Background = 0
        
        # Random peaks simulating layers (e.g. RPE, NFL)
        num_peaks = np.random.randint(3, 7)
        for i in range(num_peaks):
            center = np.random.uniform(0.1, 0.9)
            width = np.random.uniform(0.01, 0.05)
            amp = np.random.uniform(0.5, 1.0)
            
            gauss = amp * np.exp(-(x - center)**2 / (2 * width**2))
            signal += gauss
            
            # Simple threshold for mask
            mask[gauss > (0.5 * amp)] = (i % 3) + 1 # Classes 1, 2, 3
            
        noise = np.random.normal(0, 0.05, self.signal_len)
        signal += noise
        signal = np.clip(signal, 0, 1).astype(np.float32)
        
        return torch.tensor(signal).unsqueeze(0), torch.tensor(mask) # [1, L], [L]

class OCTBScanDataset(Dataset):
    """ 
    Dataset that loads B-scan images and extracts A-scans (columns).
    Assumes B-scans are images where Height = A-scan Length, Width = Number of A-scans.
    """
    def __init__(self, root_dir, signal_len=None, transform=None, samples_per_image=50, extension="*.jpeg"):
        """
        root_dir: Directory containing images (recursive search).
        signal_len: Resize A-scan to this length. If None, use original height.
        samples_per_image: Number of random A-scans to extract per image per epoch.
        """
        self.files = glob(os.path.join(root_dir, "**", extension), recursive=True)
        if len(self.files) == 0:
            # try jpg, png, tif
            self.files = glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True) + \
                         glob(os.path.join(root_dir, "**", "*.png"), recursive=True) + \
                         glob(os.path.join(root_dir, "**", "*.tif"), recursive=True)
        
        print(f"Found {len(self.files)} images.")
        self.signal_len = signal_len
        self.samples_per_image = samples_per_image
        self.transform = transform
        
        # Cache to speed up sequential access of same image
        self.cached_img_idx = -1
        self.cached_image = None

    def __len__(self):
        return len(self.files) * self.samples_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_image
        
        if img_idx != self.cached_img_idx:
            try:
                # Load image
                img_path = self.files[img_idx]
                image = Image.open(img_path).convert('L') # [Width, Height] in PIL term, but actually [W, H]
                image = np.array(image) # [H, W] usually for numpy from PIL
                
                # Check orientation: OCT usually H (depth) < W (lateral) or vice versa?
                # Usually Depth is 512-1024, Width is 512-1000.
                # If H < W, it's likely [H, W]. A-scan is column.
                
                # Normalize 0-1
                image = image.astype(np.float32) / 255.0
                self.cached_image = image
                self.cached_img_idx = img_idx
            except Exception as e:
                print(f"Error loading {self.files[img_idx]}: {e}")
                # return random noise
                return torch.zeros(1, self.signal_len or 1024)

        image = self.cached_image
        H, W = image.shape
        
        # Sample a random column
        col_idx = np.random.randint(0, W)
        ascan = image[:, col_idx] # [H]
        
        # Resize if needed
        if self.signal_len and len(ascan) != self.signal_len:
            # Simple linear interpolation
            indices = np.linspace(0, len(ascan)-1, self.signal_len)
            ascan = np.interp(indices, np.arange(len(ascan)), ascan)
            
        ascan = ascan.astype(np.float32)
        
        # Dummy mask for pretraining datasets (all zeros)
        mask = torch.zeros(self.signal_len or len(ascan), dtype=torch.long)
        
        return torch.tensor(ascan).unsqueeze(0), mask # [1, L], [L]
