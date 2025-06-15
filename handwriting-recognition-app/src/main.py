from PIL import Image
from detection.word_detector import prepare_img, detect, sort_line
from models.transformer import HandwritingTransformer
from training.trainer import train
from training.validator import ValidatorWithOutputs
from utils.params import Params
from utils.history import TrainingHistory
from data.dataset import H5Dataset, HandWritingDataset
from torch.utils.data import DataLoader
from training.early_stopping import EarlyStopping
import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from utils.progress_bar import ProgressBar # Import ProgressBar
import cv2
import threading
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Process detections through the handwriting recognition model
def predict_from_detections(detections, model, processor):
    model.eval()
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for det in detections:
            image = Image.fromarray(det.img)
            pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred_text = generated_text.strip()
            results.append((det.bbox, pred_text))
    
    return results


def capture_video(output_file='output.mp4', frame_width=640, frame_height=480, fps=20.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchvision import transforms
    import cv2

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten').to(device)

    cam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps,
                          (frame_width, frame_height))

    # Shared variable for predictions
    shared_predictions = None
    lock = threading.Lock()

    def prediction_worker():
        nonlocal shared_predictions, detections_to_process
        while True:
            with lock:
                if detections_to_process is None:  # Exit signal
                    break
                detections = detections_to_process

            # Run predictions
            predictions = predict_from_detections(detections, model, processor)

            with lock:
                shared_predictions = predictions

    # Start the prediction thread
    detections_to_process = []
    prediction_thread = threading.Thread(target=prediction_worker)
    prediction_thread.start()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        out.write(frame)
        img, binary = prepare_img(frame, frame.shape[0])
        detections = detect(frame, img, kernel_size=3, sigma=1, theta=7, min_area=500)

        # Update detections for the worker thread
        with lock:
            detections_to_process = detections

        # Display results with bounding boxes and predicted text
        with lock:
            if shared_predictions is not None:
                for bbox, text in shared_predictions:
                    x1, y1, w, h = bbox.x, bbox.y, bbox.w, bbox.h
                    cv2.rectangle(frame, (x1, y1),
                                  (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Camera', binary)
        cv2.imshow('Processed Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # Signal the prediction thread to exit
    with lock:
        detections_to_process = None
    prediction_thread.join()

    cam.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    capture_video()

if __name__ == "__main__":
    main()