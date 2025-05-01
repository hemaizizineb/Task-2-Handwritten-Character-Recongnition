import cv2
import numpy as np
import os

def load_and_preprocess_images(path):
    """
    Load and preprocess images from a directory.

    Args:
        path (str): Path to the directory containing images.

    Returns:
        tuple: A tuple containing the preprocessed images and their corresponding labels.
    """
    images = []
    labels = []
    dir_list = os.listdir(path)
    for label in dir_list:
        folder_path = os.path.join(path, label)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)