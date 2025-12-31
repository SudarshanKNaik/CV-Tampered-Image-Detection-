import pickle
import numpy as np
import os

from utils.preprocessing import preprocess_image
from features.color_stats import color_statistics
from features.edge_artifacts import edge_features
from features.noise_patterns import noise_features

# ----------------------------
# Load trained anomaly model
# ----------------------------
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

def extract_features(img):
    return np.concatenate([
        color_statistics(img),
        edge_features(img),
        noise_features(img)
    ])

# Directory containing test images
TEST_DIR = "data/raw/test"

print("\n🔍 Running tampering detection...\n")

for file in os.listdir(TEST_DIR):
    path = os.path.join(TEST_DIR, file)

    # Skip non-files (folders, etc.)
    if not os.path.isfile(path):
        continue

    # Preprocess image safely
    img = preprocess_image(path)
    if img is None:
        continue

    # Feature extraction
    feat = extract_features(img).reshape(1, -1)

    # Prediction
    pred = model.predict(feat)[0]

    # Anomaly confidence score (lower = more anomalous)
    score = model.model.decision_function(feat)[0]

    if pred == -1:
        print(f"⚠️  Manipulated | score={score:.4f} | {file}")
    else:
        print(f"✅ Clean       | score={score:.4f} | {file}")
