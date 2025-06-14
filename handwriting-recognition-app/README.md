# Handwriting Recognition App

This project is a handwriting recognition application that utilizes deep learning techniques to detect and recognize handwritten text from images. The application is structured into various modules for data handling, model training, detection, and utility functions.

## Project Structure

```
handwriting-recognition-app
├── src
│   ├── data
│   │   ├── dataset.py         # Dataset class for loading and processing handwritten text data
│   │   ├── utils.py           # Utility functions for data manipulation and preprocessing
│   │   └── transforms.py      # Image transformation functions for augmenting and preparing images
│   ├── models
│   │   ├── transformer.py      # HandwritingTransformer class for recognizing handwritten text
│   │   └── __init__.py        # Initializes the models package
│   ├── detection
│   │   ├── word_detector.py    # Functions for detecting words in images
│   │   └── __init__.py        # Initializes the detection package
│   ├── training
│   │   ├── trainer.py          # Training loop and functions for training the model
│   │   ├── validator.py        # Validation logic for evaluating model performance
│   │   └── early_stopping.py   # EarlyStopping class to prevent overfitting
│   ├── utils
│   │   ├── params.py           # Parameters and constants used throughout the project
│   │   └── history.py          # Tracks training history, including loss and accuracy
│   ├── camera
│   │   └── capture.py          # Code for capturing video from the camera
│   └── main.py                 # Entry point for the application
├── notebooks
│   └── detecting_words.ipynb    # Jupyter notebook for detecting words in images
├── data
│   ├── train                    # Directory for training images
│   ├── val                      # Directory for validation images
│   └── test                     # Directory for test images
├── weights
│   └── model_best.pth          # Best weights of the trained model
├── requirements.txt             # Lists dependencies required for the project
├── setup.py                     # Used for packaging the application
└── README.md                    # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/handwriting_recognition.git
   cd handwriting_recognition
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To train the model, run:
   ```
   python src/main.py --train
   ```

2. To validate the model, run:
   ```
   python src/main.py --validate
   ```

3. To perform inference on images, run:
   ```
   python src/main.py --infer
   ```

4. For real-time video capture, use:
   ```
   python src/camera/capture.py
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.