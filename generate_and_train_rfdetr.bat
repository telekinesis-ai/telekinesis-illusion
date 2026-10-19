@echo off
setlocal

rem Edit these settings. An optional first argument supplies the run/dataset name.
set "RUN_NAME=gearwheel_synthetic_%RANDOM%%RANDOM%"
if not "%~1"=="" set "RUN_NAME=%~1"
set "ILLUSION_REPO=%~dp0"
set "IRIS_REPO=%~dp0..\telekinesis-iris"
if not defined CONDA_BAT set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
set "SPEC_FILE=%ILLUSION_REPO%configs\example_bin_picking_gearwheel_2.yaml"
set "OUTPUT_DIR=%ILLUSION_REPO%output"
set "NUM_IMAGES=20"
set "SHARD_SIZE=10"
set "VARIANT=seg-medium"
set "EPOCHS=10"
set "BATCH_SIZE=8"
set "GRADIENT_ACCUMULATION_STEPS=4"
set "NUM_WORKERS=2"
set "LEARNING_RATE=0.0001"
set "ENCODER_LEARNING_RATE=0.00015"
set "EVALUATION_INTERVAL=5"
set "CHECKPOINT_INTERVAL=10"
set "SEED=42"

if not exist "%CONDA_BAT%" (
    echo Conda not found. Set CONDA_BAT to your condabin\conda.bat path. 1>&2
    exit /b 1
)
if not exist "%IRIS_REPO%\examples\train_rfdetr.py" (
    echo Training script not found. Check IRIS_REPO. 1>&2
    exit /b 1
)
if exist "%IRIS_REPO%\results\%RUN_NAME%" (
    echo Training run already exists. Choose a new RUN_NAME. 1>&2
    exit /b 1
)

echo Generating dataset: %RUN_NAME%
pushd "%ILLUSION_REPO%" || exit /b 1
set "PYTHONPATH=%ILLUSION_REPO%src"
call "%CONDA_BAT%" run --no-capture-output -n telekinesis-illusion python -u examples\generate_synthetic_data_with_bin_picking_worker.py ^
    --spec-file "%SPEC_FILE%" --dataset-name "%RUN_NAME%" ^
    --output-dir "%OUTPUT_DIR%" --num-images %NUM_IMAGES% ^
    --shard-size %SHARD_SIZE% --dataset-format coco --no-preview
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" goto failed

echo Training RF-DETR: %RUN_NAME%
pushd "%IRIS_REPO%" || exit /b 1
set "PYTHONPATH=%IRIS_REPO%\src"
call "%CONDA_BAT%" run --no-capture-output -n telekinesis-iris python -u examples\train_rfdetr.py ^
    --dataset "%OUTPUT_DIR%\%RUN_NAME%\coco_format" --run-name "%RUN_NAME%" ^
    --variant %VARIANT% --epochs %EPOCHS% --batch-size %BATCH_SIZE% ^
    --gradient-accumulation-steps %GRADIENT_ACCUMULATION_STEPS% ^
    --num-workers %NUM_WORKERS% --learning-rate %LEARNING_RATE% ^
    --encoder-learning-rate %ENCODER_LEARNING_RATE% ^
    --evaluation-interval %EVALUATION_INTERVAL% ^
    --checkpoint-interval %CHECKPOINT_INTERVAL% --seed %SEED%
set "EXIT_CODE=%ERRORLEVEL%"
popd
if not "%EXIT_CODE%"=="0" goto failed

echo Generation and training completed: %RUN_NAME%
exit /b 0

:failed
echo Pipeline stopped with exit code %EXIT_CODE%. See the error above. 1>&2
exit /b %EXIT_CODE%
