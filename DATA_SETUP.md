# Data Setup Guide

To train the OCT Foundation Model, you need a large collection of OCT B-scans. The model will extract A-scans (columns) from these images automatically.

## Option 1: Kermany et al. Dataset (Recommended)
This is a large dataset containing >80,000 OCT images.
1. Download from [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2) or [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Note: Check for "Retinal OCT").
   - Direct Link: [Large OCT Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)
2. Extract the zip file.
3. You should see folders like `train/CNV`, `train/NORMAL`, etc.
4. Point the training script to the extracted root folder:
   ```bash
   python train.py --data_path "C:\Path\To\OCT2017" --mode pretrain
   ```

## Option 2: OCTDL
[OCTDL Dataset](https://arxiv.org/abs/2212.01962) is another excellent source.

## Option 3: Custom Data
Put your `.jpg`, `.png`, or `.tif` B-scan images in any folder structure. The dataloader searches recursively.

## Mock Data
If you just want to test the code, use the `--use_mock` flag.
```bash
python train.py --use_mock
```
