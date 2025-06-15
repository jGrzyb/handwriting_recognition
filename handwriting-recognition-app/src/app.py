import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from data.dataset import H5Dataset, HandWritingDataset
from detection.word_detector import prepare_img, detect, sort_line
from models.transformer import HandwritingTransformer
from training.early_stopping import EarlyStopping
from training.trainer import train
from training.validator import ValidatorWithOutputs
from utils.history import TrainingHistory
from utils.params import Params
from utils.progress_bar import ProgressBar


def predict_from_detections(detections, model, processor):
    model.eval()
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for det in detections:
            image = Image.fromarray(det.img)
            # print(image)
            pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred_text = generated_text.strip()
            results.append((det.bbox, pred_text))
    
    return results

class UltraKillerAPP:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing App")

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=True)
        # self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten').to(self.device)

        self.camera_on = False
        self.cap = None
        self.current_photo = None # To store the photo taken by camera
        self.image_to_predict = None  # To store the image for prediction if needed
        # --- GUI Elements ---

        # Frame for buttons
        self.button_frame = tk.Frame(master, padx=10, pady=10)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.camera_button = tk.Button(self.button_frame, text="Open Camera", command=self.toggle_camera)
        self.camera_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.take_photo_button = tk.Button(self.button_frame, text="Take Photo", command=self.take_photo, state=tk.DISABLED)
        self.take_photo_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.detect_text_button = tk.Button(self.button_frame, text="Detect Text", command=self.detect_text, state=tk.DISABLED)
        self.detect_text_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Frame for image display
        self.image_frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.image_frame.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Original Image Label
        self.original_label_text = tk.Label(self.image_frame, text="Original Image")
        self.original_label_text.grid(row=0, column=0, pady=5)
        self.original_image_label = tk.Label(self.image_frame)
        self.original_image_label.grid(row=1, column=0, padx=5, pady=5)

        # Detected Text Label
        self.processed_label_text = tk.Label(self.image_frame, text="Detected Text:")
        self.processed_label_text.grid(row=0, column=1, pady=5, sticky="w")  # Align left

        self.detected_text = tk.Text(self.image_frame, wrap=tk.WORD, height=10, width=40)
        self.detected_text.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.detected_text.config(state=tk.DISABLED)  # Make it read-only

        # Scrollbar for the text area
        self.scrollbar = tk.Scrollbar(self.image_frame, command=self.detected_text.yview)
        self.scrollbar.grid(row=1, column=2, sticky="ns")
        self.detected_text['yscrollcommand'] = self.scrollbar.set

        # Configure column weight so the text area expands
        self.image_frame.grid_columnconfigure(1, weight=1)

    def load_image(self):
        """Opens a file dialog to select an image and processes it."""
        if self.camera_on:
            self.stop_camera() # Stop camera if it's running
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not load image.")
                return
            self.display_images(image)

    def toggle_camera(self):
        """Starts or stops the camera feed."""
        if self.camera_on:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        """Initializes and starts the webcam feed."""
        self.cap = cv2.VideoCapture(0)  # 0 for default camera
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera. Make sure it's connected and not in use.")
            self.camera_on = False
            self.camera_button.config(text="Open Camera")
            self.take_photo_button.config(state=tk.DISABLED)
            return

        self.camera_on = True
        self.camera_button.config(text="Stop Camera")
        self.take_photo_button.config(state=tk.NORMAL)
        # Start a new thread for updating the camera feed to avoid freezing the GUI
        self.camera_thread = threading.Thread(target=self.update_camera_feed)
        self.camera_thread.daemon = True  # Thread will exit when main program exits
        self.camera_thread.start()

    def stop_camera(self):
        """Stops the webcam feed and releases resources."""
        self.camera_on = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_button.config(text="Open Camera")
        self.take_photo_button.config(state=tk.DISABLED)
        # Clear displayed images
        self.original_image_label.config(image='')
        self.original_image_label.image = None


    def update_camera_feed(self):
        """Continuously captures frames from the camera and displays them."""
        while self.camera_on:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Store the current frame for "Take Photo" functionality
            self.current_photo = frame.copy()

            # Display the live original feed
            self._display_frame(frame, self.original_image_label)

            time.sleep(0.01)  # Small delay to reduce CPU usage

        if self.cap:
            self.cap.release()
            self.cap = None

    def take_photo(self):
        """Takes a photo from the current camera feed and processes it."""
        if self.current_photo is not None:
            # Stop the live feed if it's running after taking a photo,
            # so the taken photo remains displayed.
            if self.camera_on:
                self.stop_camera()
            self.display_images(self.current_photo)
            self.current_photo = None # Clear the stored photo
        else:
            messagebox.showinfo("Info", "No camera feed active or photo taken yet.")

    def _display_frame(self, frame, label):
        """Helper to convert OpenCV image to Tkinter PhotoImage and display."""
        if frame is None:
            label.config(image='')
            label.image = None
            return

        # Convert image to RGB (Tkinter PhotoImage expects RGB)
        if len(frame.shape) == 3: # BGR image
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else: # Grayscale image
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        self.image_to_predict = img_rgb.copy() # Store the image for prediction if needed
        self.detect_text_button.config(state=tk.NORMAL)  # Enable the detect text button
        # Convert NumPy array to PIL Image
        img_pil = Image.fromarray(img_rgb)
        
        # Resize image for display if it's too large, maintaining aspect ratio
        max_display_width = self.image_frame.winfo_width() // 2 - 20 # Half frame width minus padding
        max_display_height = self.image_frame.winfo_height() - 60 # Frame height minus padding for text
        
        if max_display_width <= 0 or max_display_height <= 0: # Avoid division by zero or negative size
            max_display_width = 400 
            max_display_height = 300 

        img_width, img_height = img_pil.size
        
        if img_width > max_display_width or img_height > max_display_height:
            ratio = min(max_display_width / img_width, max_display_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)


        # Convert PIL Image to Tkinter PhotoImage
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update the label with the new image
        label.config(image=img_tk)
        label.image = img_tk  # Keep a reference to prevent garbage collection


    def display_images(self, original_image):
        """Displays both the original and processed images."""
        if original_image is None:
            self.original_image_label.config(image='')
            self.original_image_label.image = None
            self.detected_text.config(state=tk.NORMAL)
            self.detected_text.delete("1.0", tk.END)
            self.detected_text.config(state=tk.DISABLED)
            return

        # Display original image
        self._display_frame(original_image, self.original_image_label)

    def detect_text(self):
        if self.image_to_predict is None:
            messagebox.showwarning("Warning", "No image to process. Please load an image or take a photo first.")
            return
        # img = prepare_img(self.image_to_predict, self.image_to_predict.shape[0])


        data = pytesseract.image_to_data(self.image_to_predict, output_type=pytesseract.Output.DICT)
        n_boxes = len(data['text'])
        current_line = -1
        line_words = []
        text = ""
        self.display_text("")
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Ignore low-confidence words
                word = data['text'][i].strip()
                if word:
                    if data['line_num'][i] != current_line:
                        # End of the previous line
                        text += "\n"
                        current_line = data['line_num'][i]
                    else:
                        text += ' '
                    text += word

        self.display_text(text)  # Display the detected text in the text area

        return

    def display_text(self, text):
        """Displays the detected text in the text area."""
        self.detected_text.config(state=tk.NORMAL)  # Enable editing
        self.detected_text.delete("1.0", tk.END)  # Clear existing text
        self.detected_text.insert(tk.END, text)  # Insert new text
        self.detected_text.config(state=tk.DISABLED)  # Disable editing


# --- Main Application Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = UltraKillerAPP(root)
    root.geometry("1000x700") # Set initial window size
    root.mainloop()