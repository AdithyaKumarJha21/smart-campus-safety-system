# Smart Campus Safety System

This project detects three safety events from surveillance video:

- `Fall Detected`
- `Normal Activity`
- `Intrusion Detected`

OpenCV reads video frames and derives motion/edge information. MediaPipe extracts 33 human-pose landmarks per frame. A TensorFlow BiLSTM classifies 30-frame pose sequences, and the video output displays the event prediction for rapid alert prioritization.

The output includes a real-time safety state: `Normal Activity`, early `Fall Risk`, `Fall Detected`, or `Intrusion Detected`. High-priority transitions are recorded in an alert CSV beside the annotated video.

## Setup

```cmd
"C:\Users\adith\AppData\Local\Programs\Python\Python311\python.exe" -m venv C:\Users\adith\AppData\Local\SmartCampusRuntime
"C:\Users\adith\AppData\Local\SmartCampusRuntime\Scripts\python.exe" -m pip install -r requirements.txt
```

## Train the BiLSTM

The retained UR dataset is stored in `data/raw/archive/UR_fall_detection_dataset_cam0_rgb`.

```cmd
"C:\Users\adith\AppData\Local\SmartCampusRuntime\Scripts\python.exe" src/data/process_ur_dataset.py
"C:\Users\adith\AppData\Local\SmartCampusRuntime\Scripts\python.exe" src/models/train_real_data.py
```

Training saves the model as `models/BiLSTM_best.keras`, normalization statistics in `models/sequence_normalization.npz`, and evaluation results in `notebooks/results/`.

Run both commands again after changing the dataset-processing code. This rebuilds the pose sequences, including the final frames of each fall video, before replacing the trained model.

## Run Detection

To choose a video from your device, run:

```cmd
run_visual_demo.bat
```

Select an `.mp4`, `.avi`, `.mov`, or `.mkv` file in the file picker.
Select the original video, not a previously generated file from `outputs/`; reprocessing an annotated output retains its old labels and panels.

You can also pass a video path directly:

```cmd
"C:\Users\adith\AppData\Local\SmartCampusRuntime\Scripts\python.exe" src/predict.py --video data/raw/archive/UR_fall_detection_dataset_cam0_rgb/fall-15-cam0-rgb
```

Or use one of the shortcuts:

```cmd
run_output.bat fall-15
run_output.bat adl-31
run_output.bat intrusion-09
```

Annotated videos are written to `outputs/`.
