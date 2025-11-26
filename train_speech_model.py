# train_speech_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pathlib
import librosa

# Configuration
DATASET_PATH = "speech_commands_v0_extracted"
SAMPLE_RATE = 16000
DURATION = 1  # seconds
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160
COMMANDS = ['on', 'off']
BATCH_SIZE = 32
EPOCHS = 50

print("=" * 50)
print("SPEECH COMMAND RECOGNITION TRAINING PIPELINE")
print("=" * 50)

# Download dataset
def download_dataset():
    """Download Google Speech Commands Dataset"""
    dataset_url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    data_dir = pathlib.Path(DATASET_PATH)
    
    if not data_dir.exists():
        print("\nDownloading Speech Commands Dataset...")
        tf.keras.utils.get_file(
            'speech_commands_v0.02.tar.gz',
            dataset_url,
            cache_dir='.',
            cache_subdir='.',
            extract=True
        )
    print(f"Dataset ready at: {data_dir}")
    return data_dir

# Load and preprocess audio
def load_audio_file(file_path):
    """Load audio file and convert to mel spectrogram"""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad if necessary
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)), mode='constant')
        else:
            audio = audio[:SAMPLE_RATE]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def prepare_dataset(data_dir):
    """Prepare training dataset"""
    X, y = [], []
    label_to_idx = {cmd: idx for idx, cmd in enumerate(COMMANDS)}
    
    print("\nLoading audio files...")
    for command in COMMANDS:
        command_dir = data_dir / command
        if not command_dir.exists():
            print(f"Warning: {command} directory not found, skipping...")
            continue
            
        audio_files = list(command_dir.glob('*.wav'))
        print(f"Loading {len(audio_files)} files for '{command}'...")
        
        for audio_file in audio_files[:1000]:  # Limit to 1000 per class for faster training
            mel_spec = load_audio_file(str(audio_file))
            if mel_spec is not None:
                X.append(mel_spec)
                y.append(label_to_idx[command])
    
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension for CNN
    X = X[..., np.newaxis]
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, label_to_idx

# Build lightweight CNN model for embedded deployment
def build_model(input_shape, num_classes):
    """Build lightweight CNN optimized for Raspberry Pi"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Conv block 1
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv block 2
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv block 3
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Training with performance tracking
def train_model(model, X_train, y_train, X_val, y_val):
    """Train model with detailed performance metrics"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\nTraining history saved to 'training_history.png'")

def evaluate_model(model, X_test, y_test, label_to_idx):
    """Detailed model evaluation"""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Overall metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Per-class metrics
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    print("\nPer-Class Performance:")
    for idx, label in idx_to_label.items():
        mask = y_test == idx
        if mask.sum() > 0:
            class_acc = (pred_classes[mask] == y_test[mask]).mean()
            print(f"  {label}: {class_acc:.4f} ({mask.sum()} samples)")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    return test_acc, test_loss

def convert_to_tflite(model, model_name='speech_model'):
    """Convert Keras model to TFLite for Raspberry Pi"""
    print("\n" + "=" * 50)
    print("CONVERTING TO TFLITE")
    print("=" * 50)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save model
    tflite_path = f'{model_name}.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = os.path.getsize(tflite_path) / 1024
    print(f"\nTFLite model saved: {tflite_path}")
    print(f"Model size: {model_size:.2f} KB")
    
    return tflite_path

# Main training pipeline
def main():
    # Download and prepare data
    data_dir = download_dataset()
    X, y, label_to_idx = prepare_dataset(pathlib.Path(DATASET_PATH))
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build and train model
    model = build_model(X_train.shape[1:], len(COMMANDS))
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot results
    plot_training_history(history)
    
    # Evaluate
    test_acc, test_loss = evaluate_model(model, X_test, y_test, label_to_idx)
    
    # Convert to TFLite
    tflite_path = convert_to_tflite(model)
    
    # Save label mapping
    import json
    with open('labels.json', 'w') as f:
        json.dump(label_to_idx, f)
    print("\nLabel mapping saved to 'labels.json'")
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"\nFiles to transfer to Raspberry Pi:")
    print(f"  - {tflite_path}")
    print(f"  - labels.json")

if __name__ == "__main__":
    main()
