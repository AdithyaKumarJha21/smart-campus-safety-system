import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from baseline_models import build_bilstm_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def plot_training_history(history, output_path):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(output_path)
    plt.close()

def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}
    print(f"Class weights: {class_weights}")
    return class_weights


def normalize_splits(X_train, X_val, X_test, output_path="models/sequence_normalization.npz"):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Saved normalization stats: {output_path}")

    return (
        ((X_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
    )


def augment_minority_classes(X_train, y_train, target_count=None):
    counts = np.bincount(y_train, minlength=3)
    if target_count is None:
        target_count = int(counts.max())

    augmented_X = [X_train]
    augmented_y = [y_train]

    for label, count in enumerate(counts):
        needed = target_count - count
        if needed <= 0:
            continue

        label_indices = np.where(y_train == label)[0]
        sampled_indices = np.random.choice(label_indices, size=needed, replace=True)
        samples = X_train[sampled_indices].copy()

        noise = np.random.normal(0, 0.03, size=samples.shape).astype(np.float32)
        scale = np.random.uniform(0.95, 1.05, size=(needed, 1, 1)).astype(np.float32)
        samples = samples * scale + noise

        for sample in samples:
            shift = np.random.randint(-2, 3)
            if shift:
                sample[:] = np.roll(sample, shift=shift, axis=0)

        augmented_X.append(samples.astype(np.float32))
        augmented_y.append(np.full(needed, label, dtype=y_train.dtype))

    X_balanced = np.concatenate(augmented_X, axis=0)
    y_balanced = np.concatenate(augmented_y, axis=0)
    shuffle_indices = np.random.permutation(len(y_balanced))

    print(f"Training labels before augmentation: {counts}")
    print(f"Training labels after augmentation:  {np.bincount(y_balanced, minlength=3)}")
    return X_balanced[shuffle_indices], y_balanced[shuffle_indices]


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=60, batch_size=16):
    print("\n--- Training BiLSTM ---")
    
    # Checkpoints and callbacks
    checkpoint_path = "models/BiLSTM_best.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, mode='max', restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    class_weights = get_class_weights(y_train)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate
    print("\n--- Evaluating BiLSTM ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
    
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    classes = ['fall', 'normal', 'intrusion']
    print(classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    ))
    
    # Save plots
    os.makedirs("notebooks/results", exist_ok=True)
    plot_training_history(history, "notebooks/results/BiLSTM_history.png")
    plot_confusion_matrix(y_test, y_pred, classes, "notebooks/results/BiLSTM_cm.png")
    return {
        "model": "BiLSTM",
        "test_accuracy": float(accuracy),
        "test_loss": float(loss),
    }

def save_training_summary(result, output_path="notebooks/results/training_results.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        f = open(output_path, "w", encoding="utf-8")
    except PermissionError:
        output_path = "notebooks/results/training_results_backup.csv"
        print(f"Warning: training result file is not writable. Saving to {output_path} instead.")
        f = open(output_path, "w", encoding="utf-8")

    with f:
        f.write("model,test_accuracy,test_loss\n")
        f.write(f"{result['model']},{result['test_accuracy']:.4f},{result['test_loss']:.4f}\n")
    print(f"Saved training summary: {output_path}")

def split_sequence_dataset(X, y, groups=None):
    if groups is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    unique_groups = np.unique(groups)
    if len(unique_groups) < 10:
        print("Warning: not enough source videos for group-aware split. Using stratified random split.")
        return split_sequence_dataset(X, y, groups=None)

    print("Using group-aware split: sequences from the same video stay in one split.")
    test_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_val_indices, test_indices = next(test_splitter.split(X, y, groups))

    X_train_val = X[train_val_indices]
    y_train_val = y[train_val_indices]
    groups_train_val = groups[train_val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    val_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    train_indices, val_indices = next(
        val_splitter.split(X_train_val, y_train_val, groups_train_val)
    )

    return (
        X_train_val[train_indices],
        X_train_val[val_indices],
        X_test,
        y_train_val[train_indices],
        y_train_val[val_indices],
        y_test,
    )


def load_real_dataset(processed_path="data/processed/ur_fall_dataset/processed_sequences.npz"):
    """Load processed UR Fall Detection Dataset."""
    if os.path.exists(processed_path):
        print(f"Loading real dataset from {processed_path}...")
        data = np.load(processed_path)
        X = data['X']
        y = data['y']
        groups = data["groups"] if "groups" in data.files else None
        print(f"Loaded {len(X)} sequences")
        print(f"  Shape: {X.shape}, Labels: {np.bincount(y)}")
        if groups is None:
            print("  Groups: not found. Re-process the dataset for stricter video-level evaluation.")
        else:
            print(f"  Source videos: {len(np.unique(groups))}")
        return X, y, groups
    else:
        print(f"ERROR: Dataset not found at {processed_path}")
        print("   Run these commands first:")
        print("   1. python src/data/download_kaggle_dataset.py")
        print("   2. python src/data/process_ur_dataset.py")
        return None, None, None


def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence models on processed sequence data.")
    parser.add_argument("--processed-path", default="data/processed/ur_fall_dataset/processed_sequences.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X_seq, y_seq, groups = load_real_dataset(args.processed_path)
    if X_seq is None:
        raise SystemExit("Process the UR dataset before training the BiLSTM model.")

    print("\nTRAINING WITH REAL UR FALL DETECTION DATASET")
    print("=" * 60)
    
    # Split real dataset
    (
        X_seq_train,
        X_seq_val,
        X_seq_test,
        y_seq_train,
        y_seq_val,
        y_seq_test,
    ) = split_sequence_dataset(X_seq, y_seq, groups)
    X_seq_train, X_seq_val, X_seq_test = normalize_splits(X_seq_train, X_seq_val, X_seq_test)
    X_seq_train_aug, y_seq_train_aug = augment_minority_classes(X_seq_train, y_seq_train)
    
    sequence_length = X_seq_train.shape[1]
    num_keypoints = X_seq_train.shape[2]
    
    print(f"\nTraining/Validation/Test split: {X_seq_train.shape[0]} / {X_seq_val.shape[0]} / {X_seq_test.shape[0]}")
    model = build_bilstm_model(sequence_length=sequence_length, num_keypoints=num_keypoints)
    result = train_and_evaluate_model(
        model,
        X_seq_train_aug,
        y_seq_train_aug,
        X_seq_val,
        y_seq_val,
        X_seq_test,
        y_seq_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    save_training_summary(result)
    
    print("\nTraining pipeline completed successfully.")
