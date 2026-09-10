import argparse
import csv
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from mediapipe.python.solutions import pose as mp_pose
except ImportError:
    mp_pose = None


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"
CLASSES = ["Fall Detected", "Normal Activity", "Intrusion Detected"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def load_model_if_exists(model_path, required=False):
    model_path = resolve_path(model_path)
    if not model_path.exists():
        message = f"Model not found: {model_path}"
        if required:
            raise FileNotFoundError(message)
        print(f"Warning: {message}")
        return None

    print(f"Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)


def load_sequence_normalization(stats_path="models/sequence_normalization.npz"):
    stats_path = resolve_path(stats_path)
    if not stats_path.exists():
        return None

    stats = np.load(stats_path)
    return stats["mean"], stats["std"]


@dataclass
class SafetyEvent:
    label: str = "Monitoring"
    priority: str = "Routine"
    confidence: float = 0.0
    color: tuple = (0, 200, 0)


class FallRiskMonitor:
    """Detect a rapid posture change while the BiLSTM gathers temporal context."""

    SHOULDER_IDS = (11, 12)
    HIP_IDS = (23, 24)
    ANKLE_IDS = (27, 28)

    def __init__(self):
        self.hip_history = deque(maxlen=12)
        self.angle_history = deque(maxlen=5)
        self.horizontal_frames = 0

    @staticmethod
    def _midpoint(keypoints, landmark_ids):
        points = [keypoints[index * 3:index * 3 + 2] for index in landmark_ids]
        if any(np.allclose(point, 0.0) for point in points):
            return None
        return np.mean(points, axis=0)

    def update(self, keypoints):
        shoulders = self._midpoint(keypoints, self.SHOULDER_IDS)
        hips = self._midpoint(keypoints, self.HIP_IDS)
        ankles = self._midpoint(keypoints, self.ANKLE_IDS)
        if shoulders is None or hips is None or ankles is None:
            self.horizontal_frames = 0
            return False, 0.0

        torso = hips - shoulders
        torso_angle = float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1]) + 1e-6)))
        body_height = float(max(ankles[1] - shoulders[1], 1e-3))
        self.hip_history.append(float(hips[1]))
        self.angle_history.append(torso_angle)

        hip_drop = 0.0
        if len(self.hip_history) >= 8:
            hip_drop = max(0.0, self.hip_history[-1] - self.hip_history[0])

        horizontal = torso_angle >= 45.0
        self.horizontal_frames = self.horizontal_frames + 1 if horizontal else 0
        rapid_drop = hip_drop >= max(0.06, body_height * 0.20)
        high_tilt = torso_angle >= 32.0
        risk = (rapid_drop and high_tilt) or self.horizontal_frames >= 3
        score = min(1.0, (torso_angle / 70.0) * 0.65 + (hip_drop / 0.20) * 0.35)
        return risk, score


def put_label(frame, text, y, color=(255, 255, 255)):
    cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def frame_paths_from_directory(video_dir):
    video_dir = resolve_path(video_dir)
    return sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def select_video_from_device():
    """Open a native file picker for a user-supplied test video."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_path = filedialog.askopenfilename(
            title="Select a video for Smart Campus Safety detection",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        root.destroy()
        return Path(selected_path) if selected_path else None
    except Exception as exc:
        raise RuntimeError(
            "Could not open the video picker. Pass --video with a video path instead."
        ) from exc


class PoseExtractor:
    def __init__(self):
        self.pose = None
        pose_module = None
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            pose_module = mp.solutions.pose
        elif mp_pose is not None:
            pose_module = mp_pose

        if pose_module is not None:
            self.pose = pose_module.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
            )
        else:
            print("Warning: MediaPipe pose is unavailable. Using motion/edge features for sequence prediction.")

    def extract(self, frame):
        if self.pose is None:
            return np.zeros(33 * 3, dtype=np.float32), None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return np.zeros(33 * 3, dtype=np.float32), None

        keypoints = []
        xs = []
        ys = []
        h, w = frame.shape[:2]

        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
            xs.append(int(landmark.x * w))
            ys.append(int(landmark.y * h))

        x1, x2 = max(0, min(xs)), min(w - 1, max(xs))
        y1, y2 = max(0, min(ys)), min(h - 1, max(ys))
        return np.array(keypoints, dtype=np.float32), (x1, y1, x2, y2)


def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 80, 160)


def detect_motion(frame, background_subtractor):
    mask = background_subtractor.apply(frame)
    mask = cv2.medianBlur(mask, 5)
    _, thresh = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 800:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h, area))

    boxes.sort(key=lambda item: item[4], reverse=True)
    return thresh, boxes[:5]


def build_motion_edge_features(frame, edge_frame, motion_mask, motion_boxes):
    h, w = frame.shape[:2]
    if motion_boxes:
        x1, y1, x2, y2, area = motion_boxes[0]
    else:
        x1 = y1 = x2 = y2 = area = 0

    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    base_features = np.array(
        [
            x1 / w,
            y1 / h,
            x2 / w,
            y2 / h,
            (x1 + x2) / (2 * w) if box_w else 0,
            (y1 + y2) / (2 * h) if box_h else 0,
            area / float(w * h),
            box_w / w,
            box_h / h,
            (box_w / box_h) if box_h else 0,
            float(np.mean(edge_frame > 0)),
            float(np.mean(motion_mask > 0)),
        ],
        dtype=np.float32,
    )

    repeats = int(np.ceil((33 * 3) / len(base_features)))
    return np.tile(base_features, repeats)[: 33 * 3].astype(np.float32)


def draw_video_annotations(frame, edge_frame, motion_mask, motion_boxes, pose_box, event):
    annotated = frame.copy()

    for x1, y1, x2, y2, _ in motion_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if pose_box is not None:
        x1, y1, x2, y2 = pose_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.rectangle(annotated, (12, 12), (430, 92), (18, 18, 18), -1)
    cv2.rectangle(annotated, (12, 12), (430, 92), event.color, 2)
    put_label(annotated, event.label, 42, event.color)
    put_label(annotated, f"Priority: {event.priority}   Confidence: {event.confidence * 100:.0f}%", 72, (245, 245, 245))

    edge_bgr = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)
    put_label(edge_bgr, "Edge Detection", 28, (0, 255, 255))

    motion_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    put_label(motion_bgr, "Motion Mask", 28, (0, 255, 255))

    side_w = annotated.shape[1] // 2
    side_h = annotated.shape[0] // 2
    edge_panel = cv2.resize(edge_bgr, (side_w, side_h))
    motion_panel = cv2.resize(motion_bgr, (side_w, side_h))
    side_panel = np.vstack([edge_panel, motion_panel])
    annotated_resized = cv2.resize(annotated, (annotated.shape[1], side_panel.shape[0]))
    return np.hstack([annotated_resized, side_panel])


def select_safety_event(seq_pred, fall_risk_active, fall_risk_score, fall_detected_latched):
    if fall_detected_latched or (seq_pred[0] == 0 and seq_pred[1] >= 0.55):
        confidence = max(seq_pred[1], 0.55) if seq_pred[0] == 0 else 1.0
        return SafetyEvent("FALL DETECTED", "Critical", confidence, (0, 0, 255))
    if fall_risk_active:
        return SafetyEvent("FALL RISK", "High", fall_risk_score, (0, 165, 255))
    if seq_pred[0] == 2 and seq_pred[1] >= 0.55:
        return SafetyEvent("INTRUSION DETECTED", "High", seq_pred[1], (255, 0, 0))
    if seq_pred[0] == 1:
        return SafetyEvent("NORMAL ACTIVITY", "Routine", seq_pred[1], (0, 200, 0))
    return SafetyEvent()


def write_alert(alert_path, frame_number, fps, event):
    new_file = not alert_path.exists()
    with alert_path.open("a", newline="", encoding="utf-8") as alert_file:
        writer = csv.writer(alert_file)
        if new_file:
            writer.writerow(["frame", "time_seconds", "event", "priority", "confidence"])
        writer.writerow([
            frame_number,
            f"{frame_number / max(fps, 1):.2f}",
            event.label,
            event.priority,
            f"{event.confidence:.4f}",
        ])


def iter_video_frames(source):
    source = resolve_path(source)
    if source.is_dir():
        for frame_path in frame_paths_from_directory(source):
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                yield frame
        return

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def source_fps(source, fallback=15):
    source = resolve_path(source)
    if source.is_dir():
        return fallback
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 1:
        return fps
    return fallback


def run_video_demo(
    video_source,
    sequence_model_path="models/BiLSTM_best.keras",
    show=True,
    max_frames=0,
):
    video_source = resolve_path(video_source)
    if video_source.parent.resolve() == OUTPUT_DIR.resolve() and "_output" in video_source.stem:
        print("Warning: this appears to be an annotated output video. Select the original source video for clean detection.")
    sequence_model = load_model_if_exists(sequence_model_path, required=True)
    sequence_norm = load_sequence_normalization()

    pose_extractor = PoseExtractor()
    sequence_length = 30
    sequence_buffer = deque(maxlen=sequence_length)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=120,
        varThreshold=40,
        detectShadows=True,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    source_name = video_source.name if video_source.is_file() else video_source.name
    output_path = OUTPUT_DIR / f"{Path(source_name).stem}_video_output.mp4"
    if video_source.is_file() and output_path.resolve() == video_source.resolve():
        output_path = OUTPUT_DIR / f"{Path(source_name).stem}_annotated.mp4"
    alert_path = OUTPUT_DIR / f"{Path(source_name).stem}_alerts.csv"
    if alert_path.exists():
        alert_path.unlink()
    writer = None
    last_seq_pred = (None, 0.0)
    last_alert_label = None
    frames_processed = 0
    video_fps = source_fps(video_source)
    fall_risk_monitor = FallRiskMonitor()
    fall_risk_until = -1
    held_fall_risk_score = 0.0
    fall_detected_latched = False

    for frame in iter_video_frames(video_source):
        if max_frames and frames_processed >= max_frames:
            break

        frame = cv2.resize(frame, (640, 480))
        edges = detect_edges(frame)
        motion_mask, motion_boxes = detect_motion(frame, background_subtractor)
        keypoints, pose_box = pose_extractor.extract(frame)
        if pose_box is None and motion_boxes:
            x1, y1, x2, y2, _ = motion_boxes[0]
            pose_box = (x1, y1, x2, y2)
        if not np.any(keypoints):
            keypoints = build_motion_edge_features(frame, edges, motion_mask, motion_boxes)
            fall_risk = False
            fall_risk_score = 0.0
        else:
            fall_risk, fall_risk_score = fall_risk_monitor.update(keypoints)
        if fall_risk:
            fall_risk_until = frames_processed + int(video_fps * 2)
            held_fall_risk_score = fall_risk_score
        sequence_buffer.append(keypoints)

        if sequence_model is not None and len(sequence_buffer) == sequence_length:
            sequence = np.expand_dims(np.array(sequence_buffer, dtype=np.float32), axis=0)
            if sequence_norm is not None:
                mean, std = sequence_norm
                sequence = ((sequence - mean) / std).astype(np.float32)
            preds = sequence_model.predict(sequence, verbose=0)
            class_idx = int(np.argmax(preds[0]))
            confidence = float(preds[0][class_idx])
            last_seq_pred = (class_idx, confidence)

        if last_seq_pred[0] == 0 and last_seq_pred[1] >= 0.55:
            fall_detected_latched = True
        fall_risk_active = frames_processed <= fall_risk_until
        event = select_safety_event(
            last_seq_pred,
            fall_risk_active,
            held_fall_risk_score if fall_risk_active else fall_risk_score,
            fall_detected_latched,
        )
        if event.label != last_alert_label and event.priority != "Routine":
            write_alert(alert_path, frames_processed, video_fps, event)
        last_alert_label = event.label

        output_frame = draw_video_annotations(
            frame,
            edges,
            motion_mask,
            motion_boxes,
            pose_box,
            event,
        )

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                video_fps,
                (output_frame.shape[1], output_frame.shape[0]),
            )

        writer.write(output_frame)
        frames_processed += 1

        if show:
            if not safe_imshow("Video Output - Motion + Edge + Detection", output_frame, delay=1):
                show = False

    if writer is not None:
        writer.release()

    if show:
        cv2.destroyAllWindows()

    print(f"Video frames processed: {frames_processed}")
    print(f"Saved video output: {output_path}")
    if alert_path.exists():
        print(f"Saved alert log: {alert_path}")
    return output_path


def safe_imshow(window_name, frame, delay=0):
    try:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay)
        return key != ord("q")
    except cv2.error:
        print("OpenCV display window is unavailable here; saved output files instead.")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Campus Safety video surveillance demo."
    )
    parser.add_argument("--video", default=None, help="Video file or frame-folder path. Opens a picker when omitted.")
    parser.add_argument("--sequence-model", default="models/BiLSTM_best.keras")
    parser.add_argument("--no-show", action="store_true", help="Save outputs without opening display windows.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum video frames to process; 0 processes the full video.")
    return parser.parse_args()


def main():
    args = parse_args()

    video_source = resolve_path(args.video) if args.video else select_video_from_device()

    if video_source is None:
        print("No video selected.")
    else:
        run_video_demo(
            video_source,
            sequence_model_path=args.sequence_model,
            show=not args.no_show,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
