# OCT A-Scan Foundation Model

This repository contains code to train a **Foundation Model for 1-Dimensional OCT A-scans**. 
The goal is to leverage large amounts of unlabeled OCT data (A-scans derived from B-scan images) to learn robust representations using Self-Supervised Learning (Masked Signal Modeling), which can then be fine-tuned for downstream tasks like:
- **Layer Segmentation** (Retina layers, choroid, etc.)
- **Classification** (Disease detection: Normal, AMD, DME, etc.)
- **Denoising / Super-resolution**

## Architecture: "ScanSAM" (Signal-Any-Model)
Inspired by SAM (Segment Anything) and MAE (Masked Autoencoders), this model treats 1D A-scans as sequences of tokens.
- **Backbone**: 1D Vision Transformer (ViT-1D).
- **Pre-training**: Masked Signal Modeling (predicting missing parts of the A-scan).
- **Fine-tuning**: Lightweight decoders for per-point classification (segmentation) or sequence classification.

## Hardware Requirements
Designed to run on modest hardware (e.g., NVIDIA P2000 4GB).
- Uses Mixed Precision (FP16).
- Efficient 1D attention mechanisms.
- Gradient accumulation for effective batch training.

## Data
We recommend using open-source OCT datasets. Since raw A-scans are rare, this codebase includes utilities to extract A-scans from standard B-scan images (PNG/JPG/TIFF).
**Recommended Datasets:**
1. **Kermany et al. (UCSD)**: [Large reliable dataset of OCT B-scans](https://data.mendeley.com/datasets/rscbjbr9sj/2)
2. **OCTDL**: [Optical Coherence Tomography DL Dataset](https://arxiv.org/abs/2212.01962)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Training (Self-Supervised Pre-training):
   ```bash
   python train.py --mode pretrain --data_path /path/to/images --batch_size 64
   ```
3. Fine-tuning (Segmentation):
   ```bash
   python train.py --mode finetune --task segmentation --data_path /path/to/masks
   ```

## Project Structure
- `src/model.py`: Transformer architecture definitions.
- `src/dataset.py`: Data loaders for extracting A-scans from B-scans.
- `train.py`: Main training loop.
