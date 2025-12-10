# raspberry_pi_inference_no_button.py
#!/usr/bin/env python3
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import librosa

# Paramètres audio / modèle
SAMPLE_RATE = 16000
RECORD_DURATION = 1.0  # 1 seconde comme dans l'entraînement
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
MODEL_PATH = "speech_model.tflite"
COMMANDS = ["on", "off", "right", "left", "up", "down"]

class SpeechCommandRecognizer:
    def __init__(self, model_path=MODEL_PATH, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate

        print("Initializing Speech Command Recognizer...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.output_index = output_details[0]["index"]
        self.input_shape = input_details[0]["shape"]

        print("Input shape:", self.input_shape)
        print("Model loaded:", model_path)
        print("Commands:", COMMANDS)
        print()

    def record_once(self, duration=RECORD_DURATION):
        num_samples = int(duration * self.sample_rate)
        print(f"Enregistrement {duration:.2f} s... Parle maintenant.")
        audio = sd.rec(
            frames=num_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()
        sf.write("test_capture.wav", audio, self.sample_rate)
        print("Taille audio capturé :", audio.shape)
        print("Échantillon moyen:", float(np.mean(audio)), "max:", float(np.max(np.abs(audio))))
        return audio

    def preprocess(self, audio_data):
        """Convert audio to mel spectrogram - EXACT MATCH avec l'entraînement"""
        
        # Pad ou truncate à exactement SAMPLE_RATE samples (16000 pour 1 sec)
        if len(audio_data) < SAMPLE_RATE:
            audio_data = np.pad(audio_data, (0, SAMPLE_RATE - len(audio_data)), mode='constant')
        else:
            audio_data = audio_data[:SAMPLE_RATE]
        
        # Compute mel spectrogram (MÊMES paramètres que l'entraînement)
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Convert to dB (MÊME référence que l'entraînement)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Add batch and channel dimensions: (40, frames) -> (1, 40, frames, 1)
        mel_spec_db = mel_spec_db[np.newaxis, :, :, np.newaxis]
        
        print(f"Feature shape: {mel_spec_db.shape}")
        print(f"Feature range: [{mel_spec_db.min():.2f}, {mel_spec_db.max():.2f}]")
        
        return mel_spec_db.astype(np.float32)

    def infer(self, features):
        self.interpreter.set_tensor(self.input_index, features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)
        return output

    def classify(self, audio):
        features = self.preprocess(audio)
        logits = self.infer(features)
        probs = np.squeeze(logits)
        
        # Afficher toutes les probabilités pour debug
        print("Probabilités brutes:", probs)
        
        idx = int(np.argmax(probs))
        if idx < len(COMMANDS):
            print(len(COMMANDS))
            return COMMANDS[idx], float(probs[idx])
        else:
            return "unknown", float(probs[idx])

    def run_cli(self):
        print("Appuie Entrée pour enregistrer une commande")
        print("Tape 'q' puis Entrée pour quitter")
        print()

        while True:
            cmd = input("[Entrée = enregistrer | q = quitter] > ").strip().lower()
            if cmd == "q":
                print("Fin.")
                break

            audio = self.record_once()
            label, score = self.classify(audio)
            print(f"Commande détectée : {label} (score {score:.3f})")
            print()

def main():
    recognizer = SpeechCommandRecognizer()
    recognizer.run_cli()

if __name__ == "__main__":
    main()