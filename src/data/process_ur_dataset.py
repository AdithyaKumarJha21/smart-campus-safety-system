import argparse
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

try:
    from mediapipe.python.solutions import pose as mp_pose
except ImportError:
    mp_pose = None

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

class URDatasetProcessor:
    def __init__(self, dataset_root="data/raw/archive/UR_fall_detection_dataset_cam0_rgb"):
        self.dataset_root = dataset_root
        pose_module = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            pose_module = mp.solutions.pose
        elif mp_pose is not None:
            pose_module = mp_pose

        self.pose = None
        if pose_module is not None:
            self.pose = pose_module.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True
            )
        else:
            print("Warning: MediaPipe pose API is unavailable. Using motion/edge features instead.")

        self.output_dir = "data/processed/ur_fall_dataset"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_motion_edge_features(self, frame, previous_frame=None):
        """Build a 99-value feature vector from motion, edges, and image statistics."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 80, 160)

        if previous_frame is None:
            diff = np.zeros_like(gray)
        else:
            prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)

        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
        else:
            x = y = bw = bh = area = 0

        base_features = np.array([
            x / w,
            y / h,
            (x + bw) / w,
            (y + bh) / h,
            (x + bw / 2) / w if bw else 0,
            (y + bh / 2) / h if bh else 0,
            area / float(w * h),
            bw / w,
            bh / h,
            (bw / bh) if bh else 0,
            float(np.mean(edges > 0)),
            float(np.mean(motion_mask > 0)),
            float(np.mean(gray) / 255.0),
            float(np.std(gray) / 255.0),
        ], dtype=np.float32)

        repeats = int(np.ceil((33 * 3) / len(base_features)))
        return np.tile(base_features, repeats)[:33 * 3].astype(np.float32)

    def extract_pose_keypoints(self, frame, previous_frame=None):
        """Extract 33 pose keypoints from a frame, or fallback motion/edge features."""
        if self.pose is None:
            return self.extract_motion_edge_features(frame, previous_frame)

        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            # Extract 33 keypoints (x, y, z coordinates) = 99 features
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            return np.array(keypoints, dtype=np.float32)
        else:
            return self.extract_motion_edge_features(frame, previous_frame)
    
    def process_video_frames(self, frames_dir, label, sequence_length=30):
        """
        Extract pose sequences from a directory of frames.
        Returns a list of sequences (each sequence has sequence_length frames).
        """
        frame_files = sorted([f for f in os.listdir(frames_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not frame_files:
            return []
        
        frame_keypoints = []
        
        # Extract keypoints from each frame
        previous_frame = None
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Warning: Could not read {frame_path}")
                continue
            
            # Resize for consistent processing
            frame = cv2.resize(frame, (640, 480))
            
            # Extract pose or fallback motion/edge features
            keypoints = self.extract_pose_keypoints(frame, previous_frame)
            frame_keypoints.append(keypoints)
            previous_frame = frame
        
        if len(frame_keypoints) == 0:
            return []
        
        # Split into overlapping sequences and always retain the last frames.
        # Fall videos often reach the floor near the end of the recording.
        sequences = []
        last_start = len(frame_keypoints) - sequence_length
        step = max(1, sequence_length // 2)
        start_indices = list(range(0, last_start + 1, step))
        if start_indices[-1] != last_start:
            start_indices.append(last_start)

        for i in start_indices:
            seq = np.array(frame_keypoints[i:i + sequence_length])
            if seq.shape[0] == sequence_length:
                sequences.append((seq, label))
        
        return sequences
    
    def process_dataset(self, sequence_length=30, max_videos_per_class=None):
        """
        Process UR Fall Detection Dataset from pre-extracted frames.
        
        Directory structure:
        dataset_root/
        - fall-01-cam0-rgb/
          - fall-01-cam0-rgb-001.png
          - fall-01-cam0-rgb-002.png
        - fall-02-cam0-rgb/
        - adl-01-cam0-rgb/
        - adl-02-cam0-rgb/
        """
        
        all_sequences = []
        all_labels = []
        all_groups = []
        
        if not os.path.exists(self.dataset_root):
            print(f"ERROR: Dataset not found at {self.dataset_root}")
            return None, None
        
        print(f"\nProcessing dataset from {self.dataset_root}...\n")
        
        # Classify videos by name
        video_dirs = sorted([d for d in os.listdir(self.dataset_root) 
                            if os.path.isdir(os.path.join(self.dataset_root, d))])
        
        # Map class names
        class_mapping = {}
        for video_dir in video_dirs:
            if video_dir.startswith('fall'):
                class_mapping[video_dir] = 0  # Fall
            elif video_dir.startswith('adl'):
                class_mapping[video_dir] = 1  # Normal
            else:
                class_mapping[video_dir] = 2  # Intrusion (if exists)
        
        # Group by class
        classes_dict = {0: {'name': 'Fall', 'videos': []},
                       1: {'name': 'Normal', 'videos': []},
                       2: {'name': 'Intrusion', 'videos': []}}
        
        for video_dir, label in class_mapping.items():
            classes_dict[label]['videos'].append(video_dir)
        
        # Process each class
        for label, class_info in classes_dict.items():
            if not class_info['videos']:
                continue
            
            class_name = class_info['name']
            video_list = class_info['videos']
            
            # Limit videos if specified
            if max_videos_per_class:
                video_list = video_list[:max_videos_per_class]
            
            print(f"Processing {class_name} videos ({len(video_list)} total)...")
            
            for video_dir in tqdm(video_list, desc=class_name):
                video_path = os.path.join(self.dataset_root, video_dir)
                sequences = self.process_video_frames(video_path, label, sequence_length)
                all_sequences.extend(sequences)
                all_labels.extend([label] * len(sequences))
                all_groups.extend([video_dir] * len(sequences))
        
        if len(all_sequences) == 0:
            print("ERROR: No sequences extracted. Check dataset structure.")
            return None, None
        
        # Convert to numpy arrays
        X = np.array([seq[0] for seq in all_sequences])
        y = np.array(all_labels)
        groups = np.array(all_groups)
        
        print(f"\nExtraction complete!")
        print(f"   Total sequences: {len(X)}")
        print(f"   Shape: {X.shape}")
        print(f"   Labels: Fall={np.sum(y==0)}, Normal={np.sum(y==1)}, Intrusion={np.sum(y==2)}")
        
        # Save processed data
        save_path = os.path.join(self.output_dir, "processed_sequences.npz")
        np.savez_compressed(save_path, X=X, y=y, groups=groups)
        print(f"   Saved to: {save_path}")
        
        return X, y


def parse_args():
    parser = argparse.ArgumentParser(description="Process UR Fall frame folders into sequence features.")
    parser.add_argument("--dataset-root", default="data/raw/archive/UR_fall_detection_dataset_cam0_rgb")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--max-videos-per-class", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processor = URDatasetProcessor(dataset_root=args.dataset_root)
    X, y = processor.process_dataset(
        sequence_length=args.sequence_length,
        max_videos_per_class=args.max_videos_per_class,
    )
    
    if X is not None:
        print("\n" + "="*60)
        print("Dataset processing complete!")
        print("="*60)
        print("\nNext step: Train models with real data")
        print("Command: python src/models/train_real_data.py")
    else:
        print("\nERROR: Processing failed. Please check the dataset location.")
