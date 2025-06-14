from typing import List, Tuple
import cv2
import numpy as np

def prepare_img(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_img = cv2.resize(image, (new_width, target_height))
    return resized_img

def detect(image: np.ndarray, kernel_size: int, sigma: int, theta: int, min_area: int) -> List:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(bbox=BoundingBox(x, y, w, h), img=image[y:y+h, x:x+w]))

    return detections

def sort_line(detections: List) -> List[List]:
    detections.sort(key=lambda det: det.bbox.y)
    lines = []
    current_line = []

    for det in detections:
        if not current_line or det.bbox.y < current_line[-1].bbox.y + current_line[-1].bbox.h:
            current_line.append(det)
        else:
            lines.append(current_line)
            current_line = [det]

    if current_line:
        lines.append(current_line)

    return lines

class BoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class Detection:
    def __init__(self, bbox: BoundingBox, img: np.ndarray):
        self.bbox = bbox
        self.img = img