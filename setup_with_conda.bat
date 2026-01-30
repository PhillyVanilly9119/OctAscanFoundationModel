@echo off
setlocal
echo ===========================================
echo OCT Foundation Model - Robust Conda Setup
echo ===========================================

REM 1. Define Environment Name
set ENV_NAME=oct_foundation

REM 2. Check if Conda is reachable
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 'conda' not found. Please run this from valid Anaconda Prompt.
    pause
    exit /b
)

echo.
echo [1/4] Configuring Conda Environment '%ENV_NAME%'...
REM Check if env exists by trying to create it (will skip if matches)
call conda create -n %ENV_NAME% python=3.10 -y

echo.
echo [2/4] Installing Dependencies (pip inside conda)...
REM Use 'conda run' to execute pip without modifying shell activation
call conda run -n %ENV_NAME% --no-capture-output python -m pip install -r requirements.txt

echo.
echo [3/4] Running System Tests...
call conda run -n %ENV_NAME% --no-capture-output python -m unittest tests/test_model_sanity.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Tests failed.
    pause
    exit /b
)

echo.
echo [4/4] Starting Workflow...
REM Download Data
if not exist "data\OCTDL_HF" (
    echo Downloading Data...
    call conda run -n %ENV_NAME% --no-capture-output python download_hf_data.py --output_dir "data\OCTDL_HF"
)

echo Starting Training...
REM Run Training
call conda run -n %ENV_NAME% --no-capture-output python train.py --data_path "data\OCTDL_HF" --mode pretrain --batch_size 16 --epochs 5 --num_workers 0

echo.
echo ===========================================
echo Done.
echo ===========================================
pause
