import numpy as np
from scipy.stats import skew

def color_statistics(img):
    features = []
    for i in range(3):  # RGB
        channel = img[:, :, i].flatten()
        features.extend([
            np.mean(channel),
            np.std(channel),
            skew(channel)
        ])
    return np.array(features)
