# raspberry_pi_inference.py
import numpy as np
import tflite_runtime.interpreter as tflite
import pyaudio
import librosa
import json
import RPi.GPIO as GPIO
import time
from collections import deque
import threading

# Configuration
MODEL_PATH = 'speech_model.tflite'
LABELS_PATH = 'labels.json'
LED_PIN = 18  # GPIO pin for LED
BUTTON_PIN = 23  # GPIO pin for record button
SAMPLE_RATE = 16000
DURATION = 1
CHUNK = 1024
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160
CONFIDENCE_THRESHOLD = 0.7

class SpeechCommandRecognizer:
    def __init__(self):
        print("Initializing Speech Command Recognizer...")
        
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        with open(LABELS_PATH, 'r') as f:
            label_to_idx = json.load(f)
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        self.led_state = False
        GPIO.output(LED_PIN, GPIO.LOW)
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        
        print("Initialization complete!")
        print(f"Model loaded: {MODEL_PATH}")
        print(f"Commands: {list(self.idx_to_label.values())}")
        print(f"LED Pin: {LED_PIN}")
        print(f"Button Pin: {BUTTON_PIN}")
    
    def preprocess_audio(self, audio_data):
        """Convert audio to mel spectrogram"""
        # Normalize audio
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Pad or truncate
        if len(audio_data) < SAMPLE_RATE:
            audio_data = np.pad(audio_data, (0, SAMPLE_RATE - len(audio_data)))
        else:
            audio_data = audio_data[:SAMPLE_RATE]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Add batch and channel dimensions
        mel_spec_db = mel_spec_db[np.newaxis, :, :, np.newaxis]
        
        return mel_spec_db.astype(np.float32)
    
    def predict(self, audio_data):
        """Run inference on audio data"""
        # Preprocess
        input_data = self.preprocess_audio(audio_data)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_label = self.idx_to_label[predicted_idx]
        
        return predicted_label, confidence
    
    def control_led(self, command):
        """Control LED based on command"""
        if command == 'on':
            GPIO.output(LED_PIN, GPIO.HIGH)
            self.led_state = True
            print("✓ LED turned ON")
        elif command == 'off':
            GPIO.output(LED_PIN, GPIO.LOW)
            self.led_state = False
            print("✓ LED turned OFF")
    
    def record_audio(self, duration=DURATION):
        """Record audio from microphone"""
        print("\n🎤 Recording... Speak your command!")
        
        frames = []
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        for _ in range(0, int(SAMPLE_RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.int16))
        
        stream.stop_stream()
        stream.close()
        
        print("Recording complete!")
        
        audio_data = np.concatenate(frames)
        return audio_data
    
    def button_callback(self, channel):
        """Handle button press"""
        if not self.recording:
            self.recording = True
            threading.Thread(target=self.process_button_press).start()
    
    def process_button_press(self):
        """Process button press in separate thread"""
        time.sleep(0.1)  # Debounce
        
        # Record audio
        audio_data = self.record_audio()
        
        # Predict
        command, confidence = self.predict(audio_data)
        
        print(f"\nPrediction: '{command}' (confidence: {confidence:.2f})")
        
        # Control LED if confidence is high enough
        if confidence >= CONFIDENCE_THRESHOLD:
            self.control_led(command)
        else:
            print(f"⚠ Low confidence ({confidence:.2f}), ignoring command")
        
        self.recording = False
    
    def run_continuous(self):
        """Run continuous voice command recognition"""
        print("\n" + "=" * 50)
        print("VOICE COMMAND LED CONTROLLER")
        print("=" * 50)
        print("\nPress the button to record a command")
        print("Say 'on' to turn LED on, 'off' to turn LED off")
        print("Press Ctrl+C to exit\n")
        
        # Setup button interrupt
        GPIO.add_event_detect(
            BUTTON_PIN,
            GPIO.FALLING,
            callback=self.button_callback,
            bouncetime=300
        )
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            GPIO.cleanup()
            self.audio.terminate()
            print("Goodbye!")
    
    def test_mode(self):
        """Test mode - continuous recording without button"""
        print("\n" + "=" * 50)
        print("TEST MODE - Continuous Recognition")
        print("=" * 50)
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                audio_data = self.record_audio()
                command, confidence = self.predict(audio_data)
                
                print(f"Prediction: '{command}' (confidence: {confidence:.2f})")
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    self.control_led(command)
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            GPIO.cleanup()
            self.audio.terminate()

if __name__ == "__main__":
    import sys
    
    recognizer = SpeechCommandRecognizer()
    
    # Check if test mode requested
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        recognizer.test_mode()
    else:
        recognizer.run_continuous()
