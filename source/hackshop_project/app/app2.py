import os
import sounddevice as sd
import wavio
import librosa
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")  # ✅ Ensure Flask knows where templates are

# ✅ Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REFERENCE_FOLDER = os.path.join(BASE_DIR, "preprocess_audio")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
RECORDED_AUDIO_FILE = os.path.join(STATIC_FOLDER, "recorded_audio.wav")
DURATION = 5  
SAMPLE_RATE = 44100  

# ✅ Ensure directories exist
os.makedirs(REFERENCE_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")  # ✅ Fixed path to templates/index.html

@app.route("/record", methods=["POST"])
def record_audio():
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished!")

    wavio.write(RECORDED_AUDIO_FILE, audio, SAMPLE_RATE, sampwidth=2)

    # ✅ Convert Recorded Audio to Spectrogram
    y_rec, sr_rec = librosa.load(RECORDED_AUDIO_FILE, sr=None)
    spectrogram_rec = np.abs(librosa.stft(y_rec))

    best_match = None
    best_similarity = float("inf")  

    print("\n🔍 Comparing with precomputed spectrograms...")

    for folder in os.listdir(REFERENCE_FOLDER):
        folder_path = os.path.join(REFERENCE_FOLDER, folder)

        if os.path.isdir(folder_path):  
            print(f"📂 Checking category: {folder}")

            for file in os.listdir(folder_path):
                if file.endswith(".png"):  
                    file_path = os.path.join(folder_path, file)

                    # ✅ Load Reference Spectrogram Image
                    spectrogram_ref = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                    if spectrogram_ref is None:
                        print(f"❌ Could not load {file_path}")
                        continue

                    # ✅ Resize (or Crop) Spectrograms
                    min_height = min(spectrogram_rec.shape[0], spectrogram_ref.shape[0])
                    min_width = min(spectrogram_rec.shape[1], spectrogram_ref.shape[1])

                    spectrogram_rec_cropped = spectrogram_rec[:min_height, :min_width]
                    spectrogram_ref_cropped = spectrogram_ref[:min_height, :min_width]

                    # ✅ Normalize & Convert to Float
                    spectrogram_rec_cropped = cv2.normalize(spectrogram_rec_cropped, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
                    spectrogram_ref_cropped = cv2.normalize(spectrogram_ref_cropped, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

                    # ✅ Flatten Both Spectrograms for Comparison
                    spectrogram_rec_flat = spectrogram_rec_cropped.flatten()
                    spectrogram_ref_flat = spectrogram_ref_cropped.flatten()

                    # 🔹 Compute Cosine Similarity (Lower = More Similar)
                    similarity = cosine(spectrogram_rec_flat, spectrogram_ref_flat)

                    print(f"   🖼 {file} → Similarity Score: {similarity:.4f}")

                    if similarity < best_similarity:
                        best_similarity = similarity
                        best_match = folder  

    # 🔥 Step 4: Output Best-Matching Category
    if best_match:
        result = f"✅ Best Matching Category: {best_match} (Similarity: {best_similarity:.4f})"
    else:
        result = "❌ No Matching Category Found!"

    return jsonify({"message": result})

if __name__ == "__main__":
    app.run(debug=True)
