
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess the image for model input"""
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
