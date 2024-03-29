import cv2
import numpy as np
from skimage import feature
from typing import Tuple, List
from tqdm import tqdm
import mahotas

def extract_haralick_features(images):
    features = []
    for image in images:
        features_for_image = mahotas.features.haralick(image).mean(axis=0)
        features.append(features_for_image)
    return np.array(features)

def extract_lbp_features(images, radius=1, points=8):
    lbp_features = np.array([feature.local_binary_pattern(image, points, radius)
                       for image in images])
    return lbp_features.reshape(len(images), -1)  # Reshape for clarity

def features_extract_combine(images,radius=1, points=8):
    # Preprocess images (resize, binarize, etc.)

    # Extract Haralick features
    print("Extracting Haralick features...")
    haralick_features = extract_haralick_features(images)

    # Extract LBP features
    print("Extracting LBP features...")
    lbp_features = extract_lbp_features(images, radius, points)

    # Combine features (consider weighting or other techniques if needed)
    print("Combining features...")
    combined_features = np.concatenate((haralick_features, lbp_features), axis=1)

    return combined_features

