import sounddevice as sd
import wavio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# 🎤 Step 1: Record a new input audio
print("🎤 Recording... Speak now!")
duration = 5  # 5 seconds recording
sample_rate = 44100
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
print("✅ Recording finished!")

# 💾 Save the recorded file
recorded_filename = "recorded_audio.wav"
wavio.write(recorded_filename, audio, sample_rate, sampwidth=2)

# 🔍 Step 2: Load both audios (recorded & preprocessed)
recorded_audio, sr1 = librosa.load(recorded_filename, sr=None)
reference_audio, sr2 = librosa.load("reference_audio.wav", sr=None)  # Reference spectrogram file

# 🔄 Convert both to Spectrograms
recorded_spectrogram = np.abs(librosa.stft(recorded_audio))
recorded_spectrogram_db = librosa.amplitude_to_db(recorded_spectrogram, ref=np.max)

reference_spectrogram = np.abs(librosa.stft(reference_audio))
reference_spectrogram_db = librosa.amplitude_to_db(reference_spectrogram, ref=np.max)

# 📊 Plot both spectrograms
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
librosa.display.specshow(reference_spectrogram_db, sr=sr2, x_axis='time', y_axis='log')
plt.title("📌 Reference Spectrogram")
plt.colorbar()

plt.subplot(1, 2, 2)
librosa.display.specshow(recorded_spectrogram_db, sr=sr1, x_axis='time', y_axis='log')
plt.title("🎤 Recorded Spectrogram")
plt.colorbar()

plt.show()

# 🏆 Step 3: Compare Spectrograms
# Reshape both to the same size
min_shape = min(reference_spectrogram_db.shape[1], recorded_spectrogram_db.shape[1])
reference_spectrogram_db = reference_spectrogram_db[:, :min_shape]
recorded_spectrogram_db = recorded_spectrogram_db[:, :min_shape]

# 🔹 Euclidean Distance (Lower = More Similar)
euclidean_dist = np.linalg.norm(reference_spectrogram_db - recorded_spectrogram_db)
print(f"📏 Euclidean Distance: {euclidean_dist}")

# 🔹 DTW Distance (Dynamic Time Warping)
dtw_dist, _, _, _ = dtw(reference_spectrogram_db.T, recorded_spectrogram_db.T, dist=lambda x, y: np.linalg.norm(x - y))
print(f"🔄 DTW Distance: {dtw_dist}")

# ✅ Interpretation:
# - If `euclidean_dist` is **low**, the sounds are similar.
# - If `dtw_dist` is **low**, even with small timing variations, they are similar.

if euclidean_dist < 500:  
    print("✅ The recorded sound is very similar to the reference.")
elif dtw_dist < 1000:
    print("🔄 The recorded sound is similar but has some timing variations.")
else:
    print("❌ The recorded sound is different from the reference.")
