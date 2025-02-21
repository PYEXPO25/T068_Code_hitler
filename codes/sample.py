import os
import sounddevice as sd
import wavio
import librosa
import numpy as np
from scipy.spatial.distance import cosine

# 🎤 Mic Settings
duration = 10  # (Ethana neram record pannanum - Seconds)
sample_rate = 44100  # (CD-quality audio)
device_index = None  # (Default mic use pannum)

# 📁 Reference Folder Path (Inga ungaloda reference audios irukanum)
reference_folder = "hackshop_project\spectrum_audio\preprocess_audio\BURPING"
print(reference_folder)
# 🔴 Step 1: Record Audio
print("🎤 Recording start")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device_index)
sd.wait()
print("✅ Recording finished!")

# 💾 Save the recorded audio
recorded_audio_file = "recorded_audio.wav"
wavio.write(recorded_audio_file, audio, sample_rate, sampwidth=2)
print(f"💾 Audio file '{recorded_audio_file}' la save aagiduchu!")

# 🎶 Step 2: Convert Recorded Audio to Spectrogram
y_rec, sr_rec = librosa.load(recorded_audio_file, sr=None)
spectrogram_rec = np.abs(librosa.stft(y_rec)).flatten() # Flatten array for comparison


# 🔍 Step 3: Compare with Reference Folder
best_match = None
best_similarity = float("inf")  # Lower value means more similar
for folder in os.listdir(reference_folder):
    folder_path = os.path.join(reference_folder, folder)
    
    if os.path.isdir("hackshop_project\spectrum_audio\preprocess_audio\BURPING"):  # Check if it's a folder
        for file in os.listdir("hackshop_project\spectrum_audio\preprocess_audio\BURPING"):
            if file.endswith(".png"):  # Only compare .wav files
                print("here")
                file_path = os.path.join(folder_path, file)
                
                # Load reference audio
                y_ref, sr_ref = librosa.load(file_path, sr=None)
                spectrogram_ref = np.abs(librosa.stft(y_ref)).flatten()  # Flatten array
                
                # Calculate similarity (Cosine Distance)
                similarity = cosine(spectrogram_rec, spectrogram_ref)
                
                # Update best match
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_match = folder  # Save best-matching folder

# 🔥 Output Best Matching Folder
if best_match:
    print(f"✅ Matching Folder: {best_match}")
else:
    print("❌ No Matching Folder Found!")