import os
import sounddevice as sd
import wavio
import librosa
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ✅ Paths
REFERENCE_FOLDER = r"C:\Users\sabar\OneDrive\Desktop\T068_Code_hitler\hackshop_project\spectrum_audio\preprocess_audio"
RECORDED_AUDIO_FILE = "static/recorded_audio.wav"  # Save recorded audio
DURATION = 5  # Record for 5 seconds
SAMPLE_RATE = 44100  # CD-quality audio

@app.route("/")
def home():
    return render_template("index.html")  # Load frontend

@app.route("/record", methods=["POST"])
def record_audio():
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished!")

    # Save recorded audio
    wavio.write(RECORDED_AUDIO_FILE, audio, SAMPLE_RATE, sampwidth=2)

    # ✅ Step 2: Convert Recorded Audio to Spectrogram
    y_rec, sr_rec = librosa.load(RECORDED_AUDIO_FILE, sr=None)
    spectrogram_rec = np.abs(librosa.stft(y_rec))  # Do NOT flatten yet

    # ✅ Step 3: Compare with Precomputed Spectrogram Images
    best_match = None
    best_similarity = float("inf")  # Lower = More Similar

    print("\n🔍 Comparing with precomputed spectrograms...")

    for folder in os.listdir(REFERENCE_FOLDER):
        folder_path = os.path.join(REFERENCE_FOLDER, folder)

        if os.path.isdir(folder_path):  # ✅ Ensure it's a folder
            print(f"📂 Checking category: {folder}")

            for file in os.listdir(folder_path):
                if file.endswith(".png"):  # ✅ Precomputed spectrograms are in PNG format
                    file_path = os.path.join(folder_path, file)

                    # ✅ Load Reference Spectrogram Image
                    spectrogram_ref = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                    if spectrogram_ref is None:
                        print(f"❌ Could not load {file_path}")
                        continue

                    # ✅ Resize Input Spectrogram to Match Reference Image Size
                    spectrogram_rec_resized = cv2.resize(np.abs(spectrogram_rec), (spectrogram_ref.shape[1], spectrogram_ref.shape[0]))

                    # ✅ Flatten Both Spectrograms for Comparison
                    spectrogram_rec_flat = spectrogram_rec_resized.flatten()
                    spectrogram_ref_flat = spectrogram_ref.flatten()

                    # 🔹 Compute Cosine Similarity (Lower = More Similar)
                    similarity = cosine(spectrogram_rec_flat, spectrogram_ref_flat)

                    print(f"   🖼 {file} → Similarity Score: {similarity:.4f}")

                    # ✅ Update Best Match
                    if similarity < best_similarity:
                        best_similarity = similarity
                        best_match = folder  # Save Best-Matching Category

    # 🔥 Step 4: Output Best-Matching Category
    if best_match:
        result = f"✅ Best Matching Category: {best_match} (Similarity: {best_similarity:.4f})"
    else:
        result = "❌ No Matching Category Found!"

    return jsonify({"message": result})

if __name__ == "__main__":
    app.run(debug=True)
