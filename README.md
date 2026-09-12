# 🎥 Smart Campus Safety System

A computer vision system that detects falls, intrusions, and anomalies in surveillance video using pose estimation and deep learning. Built for campus safety teams to identify high-risk events in real-time, generate annotated video evidence, and log alerts for immediate response.

## ✨ Key Features

- **Fall Detection with Risk Prediction** — Detects confirmed falls and provides early "fall risk" alerts before impact occurs
- **Multi-Event Classification** — Classifies three event types: falls, normal activity, and intrusions with confidence scores
- **Pose and Motion Visualization** — Displays human pose landmarks, motion bounding boxes, and edge detection overlays on video output
- **Flexible Video Input** — Processes individual video files or folders of extracted frame sequences with interactive file picker
- **Alert Logging System** — Generates CSV logs with frame number, timestamp, event type, priority level, and model confidence
- **Annotated Video Output** — Creates marked-up surveillance videos with overlays showing detected events and risk states

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend & ML** | Python 3.11, TensorFlow/Keras 2.15.1, Scikit-learn |
| **Computer Vision** | OpenCV, MediaPipe Pose, NumPy |
| **Model Architecture** | Bidirectional LSTM (BiLSTM) |
| **Visualization & Analysis** | Matplotlib, Seaborn, Tkinter |
| **Data Processing** | tqdm, NumPy |

## 📦 Installation & Setup

### Prerequisites
- Python 3.11
- Windows OS (currently Windows-specific)
- Git

### On Windows (Command Prompt)

```cmd
git clone https://github.com/AdithyaKumarJha21/smart-campus-safety-system.git
cd smart-campus-safety-system

py -3.11 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Note
This project is currently optimized for Windows. The batch scripts and paths use Windows conventions. For macOS/Linux support, modify the batch files and use forward slashes for paths.

## 🚀 Quick Start

### Interactive video selection

```cmd
.\.venv\Scripts\python.exe src\predict.py
```

A file picker opens. Select a video file (`.mp4`, `.avi`, `.mov`, `.mkv`). Results are saved to a local `outputs/` directory.

### Process a specific video

```cmd
.\.venv\Scripts\python.exe src\predict.py ^
  --video <path-to-video> ^
  --no-show
```

### Process a folder of frames

```cmd
.\.venv\Scripts\python.exe src\predict.py ^
  --video data\processed\ur_fall_dataset ^
  --max-frames 180
```

### First Time Using It

1. After installation, run the interactive file picker command above
2. The system loads the pre-trained BiLSTM model from `models/BiLSTM_best.keras`
3. Select a video or frame folder to process
4. Video frames are processed in 30-frame sequences to detect events
5. Watch the OpenCV display window show pose landmarks, motion detection, and edge detection overlays
6. When processing completes, annotated output and alert logs are generated in a local `outputs/` directory

### Model Output Classes
Fall Detected (high priority)
FALL RISK (early warning, medium priority)
Normal Activity (low priority)
Intrusion Detected (high priority)


## 📁 Project Structure

smart-campus-safety-system/
├── README.md
├── run_output.bat
├── run_training.bat
├── run_visual_demo.bat
├── .gitignore
├── data/
│ └── processed/
│ └── ur_fall_dataset/
├── models/
│ ├── BiLSTM_best.keras
│ └── sequence_normalization.npz
└── src/
├── predict.py
├── data/
│ └── process_ur_dataset.py
└── models/
├── baseline_models.py
└── train_real_data.py