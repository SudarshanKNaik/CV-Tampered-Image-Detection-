import pickle
import numpy as np
import os
import cv2

from utils.preprocessing import preprocess_image
from features.color_stats import color_statistics
from features.edge_artifacts import edge_features
from features.noise_patterns import noise_features

# ----------------------------
# Load trained model
# ----------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# Feature extraction
# ----------------------------
def extract_features(img):
    color_feat = color_statistics(img)
    edge_feat = edge_features(img)
    noise_feat = noise_features(img)

    combined = np.concatenate([color_feat, edge_feat, noise_feat])
    return combined, color_feat, edge_feat, noise_feat

# ----------------------------
# Normalize deviation (for explanation)
# ----------------------------
def deviation_score(feature_vector):
    return np.linalg.norm(feature_vector)

# ----------------------------
# Test directory
# ----------------------------
TEST_DIR = "data/raw/test"

print("\n🔍 Image Tampering Detection Results (First 25 Images)\n")
print("-" * 110)

MAX_IMAGES = 55
count = 0

for file in os.listdir(TEST_DIR):

    if count >= MAX_IMAGES:
        break

    path = os.path.join(TEST_DIR, file)

    if not os.path.isfile(path):
        continue

    img = preprocess_image(path)
    if img is None:
        continue

    # Extract features
    feat, color_f, edge_f, noise_f = extract_features(img)
    feat = feat.reshape(1, -1)

    # Prediction
    pred = model.predict(feat)[0]
    anomaly_score = model.model.decision_function(feat)[0]

    # Explainability metrics
    color_dev = deviation_score(color_f)
    edge_strength = edge_f[0] + edge_f[1]   # density + sharpness
    noise_abnormality = abs(noise_f[1])     # noise std deviation

    label = "⚠️  Manipulated" if pred == -1 else "✅ Clean"

    print(
        f"{label:<14} | "
        f"AnomalyScore={anomaly_score:>7.3f} | "
        f"ColorDev={color_dev:>8.2f} | "
        f"EdgeInconsistency={edge_strength:>8.4f} | "
        f"NoiseAbnormality={noise_abnormality:>7.3f} | "
        f"{file}"
    )

    count += 1

print("-" * 110)
