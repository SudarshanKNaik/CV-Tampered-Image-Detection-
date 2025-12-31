import os
import numpy as np
from tqdm import tqdm

from utils.preprocessing import preprocess_image
from features.color_stats import color_statistics
from features.edge_artifacts import edge_features
from features.noise_patterns import noise_features
from models.anomaly_model import ManipulationDetector

DATA_DIR = "data/raw/train"


features = []

for file in tqdm(os.listdir(DATA_DIR)):
    path = os.path.join(DATA_DIR, file)

    img = preprocess_image(path)
    if img is None:
        continue

    f = np.concatenate([
        color_statistics(img),
        edge_features(img),
        noise_features(img)
    ])
    features.append(f)


features = np.array(features)

detector = ManipulationDetector()
detector.train(features)

import pickle
pickle.dump(detector, open("models/model.pkl", "wb"))

print("✅ Model trained (self-supervised)")
