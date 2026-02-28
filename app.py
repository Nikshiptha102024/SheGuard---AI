from flask import Flask, render_template, request
from PIL import Image, ExifTags
import numpy as np
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def analyze_image(image_path):
    img = Image.open(image_path)
    img_np = np.array(img)

    # 1️⃣ Noise Score (variance of grayscale)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    noise_variance = np.var(gray)
    noise_score = min(noise_variance / 5000, 1.0)

    # 2️⃣ Edge Score (Canny edge density)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    edge_score = min(edge_density * 5, 1.0)

    # 3️⃣ Compression Score (block artifact detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    compression_score = min(1 - (laplacian_var / 1000), 1.0)
    compression_score = max(compression_score, 0)

    # 4️⃣ Metadata Score
    metadata_score = 1.0
    try:
        exif = img._getexif()
        if exif is not None:
            metadata_score = 0.2
    except:
        metadata_score = 1.0

    # Weighted probability
    probability = (
        0.35 * noise_score +
        0.30 * edge_score +
        0.20 * compression_score +
        0.15 * metadata_score
    )

    return round(probability * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    probability = None
    risk = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            probability = analyze_image(path)

            if probability > 70:
                risk = "HIGH"
            elif probability > 40:
                risk = "MEDIUM"
            else:
                risk = "LOW"

    return render_template("index.html", probability=probability, risk=risk)

if __name__ == "__main__":
    app.run(debug=True)
