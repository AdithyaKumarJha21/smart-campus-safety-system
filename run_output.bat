@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" goto usage

set "NAME=%~1"
set "MAX_FRAMES=9999"
if not "%~2"=="" set "MAX_FRAMES=%~2"

set "SOURCE="
set "OUTPUT="
set "SEQUENCE_MODEL=models\BiLSTM_best.keras"

if /I "%NAME:~0,4%"=="adl-" (
    set "SOURCE=data\raw\archive\UR_fall_detection_dataset_cam0_rgb\%NAME%-cam0-rgb"
    set "OUTPUT=outputs\%NAME%-cam0-rgb_video_output.mp4"
    goto run
)

if /I "%NAME:~0,5%"=="fall-" (
    set "SOURCE=data\raw\archive\UR_fall_detection_dataset_cam0_rgb\%NAME%-cam0-rgb"
    set "OUTPUT=outputs\%NAME%-cam0-rgb_video_output.mp4"
    goto run
)

if /I "%NAME:~0,10%"=="intrusion-" (
    set "SOURCE=data\raw\archive\UR_fall_detection_dataset_cam0_rgb\%NAME%-cam0-rgb"
    set "OUTPUT=outputs\%NAME%-cam0-rgb_video_output.mp4"
    goto run
)

echo Unknown shortcut: %NAME%
echo.
goto usage

:run
if not exist "%SOURCE%" (
    echo Source not found:
    echo %SOURCE%
    echo.
    pause
    exit /b 1
)

echo Running output for: %NAME%
echo Source: %SOURCE%
echo Sequence model: %SEQUENCE_MODEL%
echo Max frames: %MAX_FRAMES%
echo.

"C:\Users\adith\AppData\Local\SmartCampusRuntime\Scripts\python.exe" .\src\predict.py --video "%SOURCE%" --sequence-model "%SEQUENCE_MODEL%" --max-frames %MAX_FRAMES% --no-show

if errorlevel 1 (
    echo.
    echo Prediction failed.
    pause
    exit /b 1
)

echo.
echo Opening:
echo %OUTPUT%
start "" "%OUTPUT%"
pause
exit /b 0

:usage
echo Usage:
echo   run_output.bat fall-15
echo   run_output.bat adl-31
echo   run_output.bat intrusion-01
echo.
echo Optional second argument: max frames
echo   run_output.bat fall-15 180
echo.
pause
exit /b 1
