@echo off
REM ============================================================
REM  BlockWatch Studio — one-shot model trainer
REM  Запускаешь этот .bat двойным кликом или из PowerShell.
REM  Через ~1-1.5 часа в crates/vision/models/ появится файл .onnx
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo === STEP 1/5: Installing Python dependencies ===
python -m pip install --quiet pillow ultralytics || goto :err

echo.
echo === STEP 2/5: Generating 1500 synthetic training images ===
cd training
python synth_data.py --count 1500 || goto :err

echo.
echo === STEP 3/5: Splitting dataset 90/10 train/val ===
python split_dataset.py 0.1 || goto :err

echo.
echo === STEP 4/5: Training YOLOv8n (this is the slow step, ~1 hour) ===
echo Iterations will print below. You can leave the computer alone.
python train_yolov8.py || goto :err

echo.
echo === STEP 5/5: Done! ===
echo.
echo Trained ONNX model is in:
echo   crates\vision\models\popup-yolov8n-v1.onnx
echo.
echo Next: уведоми Claude в чате "модель готова" — он подключит её к bw-vision.
echo.
pause
exit /b 0

:err
echo.
echo *** ERROR. Что-то пошло не так. Скопируй последние строки выше и пришли Claude. ***
pause
exit /b 1
