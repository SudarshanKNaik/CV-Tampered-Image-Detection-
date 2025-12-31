import cv2
import os

def preprocess_image(path, size=(256,256)):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img
