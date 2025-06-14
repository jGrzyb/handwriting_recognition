from typing import List
import numpy as np
import cv2

def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")
    return image

def save_image(image: np.ndarray, save_path: str) -> None:
    cv2.imwrite(save_path, image)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    return cv2.resize(image, size)

def normalize_image(image: np.ndarray) -> np.ndarray:
    return image / 255.0

def augment_images(images: List[np.ndarray]) -> List[np.ndarray]:
    augmented_images = []
    for img in images:
        # Example augmentation: flipping the image
        flipped = cv2.flip(img, 1)  # Horizontal flip
        augmented_images.append(flipped)
    return augmented_images