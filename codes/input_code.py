
import sounddevice as sd
import wavio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
print("Available input devices:")
print(sd.query_devices())  
duration = 5 # Secs (5 sec record pannum)
sample_rate = 44100  # CD-quality audio
device_index = None # Mic index (modify as per your system)

print("🎤 Recording... Speak now!")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device_index)
sd.wait()  # Wait till recording completes
print("✅ Recording finished!")

# Save as WAV file
wavio.write("recorded_audio.wav", audio, sample_rate, sampwidth=2)
print("📁 Audio saved as recorded_audio.wav")

y, sr = librosa.load("recorded_audio.wav", sr=None)

# 🔊 Convert to Spectrogram
spectrogram = np.abs(librosa.stft(y))
spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

# 📊 Spectrogram Display
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title("🎶 Recorded Audio Spectrogram")
plt.xlabel("⏳ Time (Seconds)")
plt.ylabel("📡 Frequency (Hz)")
plt.show()