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
            # # Convert the detected word image to PIL format
            # word_img = Image.fromarray(det.img)
            
            # # Apply the same transformations used during training
            # img_tensor = transform(word_img).unsqueeze(0).to(device)
            
            # # Start with the SOS token
            # input_token = torch.tensor([[27]]).to(device)q
            
            # # Generate tokens one by one
            # generated_tokens = [27]  # Start with SOS token
            # max_length = 20  # Set maximum word length
            
            # for _ in range(max_length):
            #     output = model(img_tensor, input_token)
            #     next_token = output[0, -1].argmax().item()
            #     generated_tokens.append(next_token)
                
            #     # Stop if we reach EOS token
            #     if next_token == 28:
            #         break
                
            #     # Update input token for next prediction
            #     input_token = torch.cat([input_token, torch.tensor([[next_token]]).to(device)], dim=1)
            
            # # Convert tokens to text
            # pred_text = Params.decode_string([i for i in generated_tokens if i not in [27, 28]])
            results.append((det.bbox, pred_text))
    
    return results


def capture_video(output_file='output.mp4', frame_width=640, frame_height=480, fps=20.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)
    
    # vocab_size = len(Params.vocab) + 2
    # model = HandwritingTransformer(
    #     input_size=16 * 24,
    #     vocab_size=len(Params.vocab)+2,
    #     d_model=128,
    #     nhead_en=1,
    #     num_layers_en=1,
    #     nhead_de=1,
    #     num_layers_de=1,
    #     dropout=0.2
    # ).to(device)
    # model.load_state_dict(torch.load(
    #     'handwriting_transformer.pth', map_location=device))
    # model.eval()
    # transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((64, 128)),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: 1.0 - x),
    #     transforms.Normalize((0.5,), (0.5,)),
    # ])

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
        img = prepare_img(frame, frame.shape[0])
        detections = detect(img, min_area=100)

        if len(detections) != 0:
            # Update detections for the worker thread
            with lock:
                detections_to_process = detections

            # Display results with bounding boxes and predicted text
            with lock:
                if shared_predictions is not None:
                    for bbox, text in shared_predictions:
                        print(f"Detected: {text} at {bbox.x}, {bbox.y}, {bbox.w}, {bbox.h}")
                        x1, y1, w, h = bbox.x, bbox.y, bbox.w, bbox.h
                        cv2.rectangle(frame, (x1, y1),
                                    (x1 + w, y1 + h), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Camera', img)

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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_dataset = H5Dataset('train_data.h5', num_epochs=1)
    # val_dataset = HandWritingDataset(root='words_data/val', transform=transform, label_transofrm=Params.encode_string)

    # train_loader = DataLoader(train_dataset, batch_size=64, num_workers=0, sampler=train_dataset.create_h5_sampler(0), collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0, shuffle=True, collate_fn=collate_fn)

    # model = HandwritingTransformer(
    #     input_size=16 * 24,
    #     vocab_size=len(Params.vocab) + 2,
    #     d_model=128,
    #     nhead_en=1,
    #     num_layers_en=1,
    #     nhead_de=1,
    #     num_layers_de=1,
    #     dropout=0.2
    # ).to(device)

    # criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # early_stopping = EarlyStopping(patience=5)
    # history = TrainingHistory()

    # train(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     epochs=1,
    #     early_stopping=early_stopping,
    #     history=history
    # )

    # # Optionally, run inference or capture video
    capture_video()

if __name__ == "__main__":
    # # Load the state_dict
    # state_dict = torch.load('D:\\JetBrains\\Projects\\handwriting_recognition\\handwriting-recognition-app\\weights\\handwriting_model.pth')

    # # Inspect the embedding.weight size
    # embedding_weight = state_dict['embedding.weight']
    # print(f"Embedding weight size: {embedding_weight.size()}")

    # # Inspect the output.weight size
    # output_weight = state_dict['output.weight']
    # print(f"Output weight size: {output_weight.size()}")

    # # Inspect the output.bias size
    # output_bias = state_dict['output.bias']
    # print(f"Output bias size: {output_bias.size()}")
    main()