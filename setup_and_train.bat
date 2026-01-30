@echo off
setlocal
cd /d "%~dp0"

echo ===========================================
echo OCT Foundation Model - Setup & Training
echo ===========================================

echo [1/4] Setting up Virtual Environment...
if not exist "venv" (
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Upgrading pip...
    python -m pip install --upgrade pip
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo [2/4] Downloading Open Source OCT Data (OCTDL/Kermany)...
if not exist "data\OCTDL_HF" (
    python download_hf_data.py --output_dir "data\OCTDL_HF"
) else (
    echo Data folder already exists. Skipping download.
)

echo.
echo [3/4] Verifying Data...
python -c "import glob; print(f'Found {len(glob.glob(\"data/OCTDL_HF/**/*.jpg\", recursive=True))} images.')"

echo.
echo [4/4] Starting Pre-training (Foundation Model)...
echo Config: ViT-Tiny, Signal Len 1024, Batch Size 16 (safe for P2000)
python train.py --data_path "data\OCTDL_HF" --mode pretrain --batch_size 16 --epochs 5 --num_workers 2

echo.
echo ===========================================
echo Training Complete! 
echo Model checkpoints are in 'models/'
echo ===========================================
pause
