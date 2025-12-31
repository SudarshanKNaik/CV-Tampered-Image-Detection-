import cv2
import numpy as np

def noise_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    noise = gray - blur

    return np.array([
        np.mean(noise),
        np.std(noise)
    ])
