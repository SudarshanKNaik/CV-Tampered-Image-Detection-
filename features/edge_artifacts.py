import cv2
import numpy as np

def edge_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges > 0) / edges.size
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return np.array([edge_density, laplacian_var])
