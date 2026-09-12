@echo off
echo Rebuilding pose sequences from the UR dataset...
python src\data\process_ur_dataset.py
if errorlevel 1 goto failed

echo.
echo Training the BiLSTM model...
python src\models\train_real_data.py
if errorlevel 1 goto failed

echo.
echo Training completed. The updated model is models\BiLSTM_best.keras
pause
exit /b 0

:failed
echo.
echo Training failed. Activate the virtual environment and confirm dependencies are installed.
pause
exit /b 1
