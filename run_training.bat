@echo off
echo Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo Starting Training with Mock Data (Test Run)...
python train.py --use_mock --epochs 5 --batch_size 32 --signal_len 1024

echo.
echo Process Complete. Check "models" folder for checkpoints.
pause
