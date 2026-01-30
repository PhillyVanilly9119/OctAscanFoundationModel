import unittest
import torch
import sys
import os

# Add parent dir to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import OctFoundationModelMAE, OctSegModel

class TestOctModel(unittest.TestCase):
    def setUp(self):
        self.signal_len = 1024
        self.batch_size = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Test Info] Testing on device: {self.device}")

    def test_foundation_model_instantiation(self):
        """Test if the MAE model initializes and runs a forward pass."""
        print("[Test] Checking Foundation Model...")
        model = OctFoundationModelMAE(signal_len=self.signal_len,
                                      embed_dim=96, depth=2, num_heads=3, # Tiny for test
                                      decoder_embed_dim=64, decoder_depth=2)
        model.to(self.device)
        
        dummy_input = torch.randn(self.batch_size, 1, self.signal_len).to(self.device)
        loss, pred, mask = model(dummy_input, mask_ratio=0.75)
        
        self.assertFalse(torch.isnan(loss).any(), "Loss contains NaNs")
        # Pred shape should be [B, L, patch_size * chans] -> actually decoding to patches
        # Output of decoder is [B, N, patch_size]
        # Check shapes
        self.assertEqual(pred.shape[0], self.batch_size)
    
    def test_segmentation_model_instantiation(self):
        """Test if the Segmentation model initializes and runs."""
        print("[Test] Checking Segmentation Model...")
        base = OctFoundationModelMAE(signal_len=self.signal_len,
                                     embed_dim=96, depth=2, num_heads=3)
        model = OctSegModel(base, num_classes=4)
        model.to(self.device)
        
        dummy_input = torch.randn(self.batch_size, 1, self.signal_len).to(self.device)
        logits = model(dummy_input)
        
        # Logits: [B, num_classes, L]
        self.assertEqual(logits.shape, (self.batch_size, 4, self.signal_len))

if __name__ == '__main__':
    unittest.main()
