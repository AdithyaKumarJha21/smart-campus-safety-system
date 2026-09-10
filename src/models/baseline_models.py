import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, LSTM, Dropout


def build_bilstm_model(sequence_length=30, num_keypoints=33 * 3, num_classes=3):
    """Classify pose-keypoint sequences with a bidirectional LSTM."""
    model = Sequential([
        tf.keras.Input(shape=(sequence_length, num_keypoints)),
        Bidirectional(LSTM(96, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(48)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
