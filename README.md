# Bangla Handwriting OCR (Isolated Characters)

This repository contains a complete pipeline for recognizing handwritten Bangla basic characters (vowels, consonants, and numerals) using the Mendeley **BanglaLekha-Isolated** dataset. It tracks experiment runs using **MLflow** and serves the models using a web-based prediction UI built with **Streamlit** (featuring a drawable canvas and character segmentation for word recognition). The Streamlit app is fully containerized using **Docker**.

---

## Project Structure

```
bangla-ocr-assignment/
├── train.py                  # PyTorch model training script with MLflow tracking
├── app.py                    # Streamlit web application with drawing board
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker packaging configuration
├── README.md                 # Project documentation
├── labels.json               # Folder ID to Bangla character mappings
├── models/
│   └── model.pkl             # Best trained PyTorch model weights (created after training)
├── artifacts/
│   └── mlflow/               # Local MLflow tracking data store
└── screenshots/              # Screenshots of web app and experiment runs
    ├── streamlit_app.png
    └── mlflow_experiment.png
```

---

## Technical Approach

### 1. Dataset & Preprocessing
The model is trained on the Mendeley **BanglaLekha-Isolated** dataset. To focus only on basic characters (excluding compound words), we train on folders `1` through `60` which contain:
- **11 Vowels** (folders 1 to 11; `অ` to `ঔ`)
- **39 Consonants & Diacritics** (folders 12 to 50; `ক` to `ঁ`)
- **10 Numerals** (folders 51 to 60; `০` to `৯`)

**Preprocessing Pipeline:**
- **Grayscale Conversion:** Input images are read as single-channel grayscale (`L` mode) using Pillow to standardize color profiles.
- **Resizing:** Images are resized to $64 \times 64$ pixels.
- **Normalization:** Pixel values are scaled to $[0.0, 1.0]$ and normalized using standard mean $= 0.5$ and standard deviation $= 0.5$.
- **Augmentation:** To increase generalization and robustness to canvas drawings, train images are augmented with random rotations (up to 10°) and affine translations (up to 5%).

### 2. Model Architectures
`train.py` supports training two different architectures to facilitate multi-experiment tracking:
1. **Custom CNN:** A fast, lightweight 3-layer Convolutional Neural Network:
   - `Conv2d(1 -> 32, 3x3)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2)`
   - `Conv2d(32 -> 64, 3x3)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2)`
   - `Conv2d(64 -> 128, 3x3)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2)`
   - `Linear(128 * 8 * 8 -> 256)` -> `Dropout(0.3)` -> `Linear(256 -> 60)`
2. **MobileNetV3 Small (Transfer Learning):** Uses a pre-trained ImageNet backbone with modified input layer (adapted to accept 1-channel grayscale) and a custom 60-class dense head.

### 3. Word Segmentation Strategy
The drawing canvas allows drawing a full word. The application segments the canvas into individual characters using **OpenCV**:
1. **Grayscale & Binarization:** Extracts the drawing alpha channel and applies binary thresholding (converting drawing strokes to white and background to black).
2. **Contour Detection:** Uses `cv2.findContours` to retrieve boundaries of individual strokes.
3. **Horizontal Stroke Merging:** Characters in Bangla often contain multiple strokes (e.g. `অ` or letters with separate loops) or modifiers that trigger multiple disjoint contours. We merge bounding boxes that overlap or are horizontally closer than a specified threshold (e.g., $25\text{px}$) to avoid over-segmentation.
4. **Horizontal Sorting:** Bounding boxes are sorted by their X coordinates from left-to-right to preserve writing order.
5. **Square Padding & Resize:** Each segmented stroke/character is cropped, padded with borders to create a square (preventing distortion during scaling), and resized to $64 \times 64$ pixels for model classification.

---

## MLflow Tracking

The training script automatically logs experiments to the local MLflow server directory (`artifacts/mlflow`). For each run, it tracks:
- **Hyperparameters:** Batch size, learning rate, epochs, model type, optimizer, seed, data counts.
- **Metrics:** Training loss, training accuracy, validation loss, and validation accuracy per epoch.
- **Plots (Artifacts):** Loss/accuracy curves and validation confusion matrices (for the first 20 classes).
- **Model Registry:** The best weights are saved locally under `models/model.pkl` and registered inside MLflow's Model Registry.

---

## Installation & Usage

### 1. Local Setup
Create the virtual environment inside the project directory and install the requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
Train the model by pointing to the dataset. You can run multiple times with different parameters or model types to create different MLflow runs:

**Run 1 (Custom CNN):**
```bash
python3 train.py --epochs 5 --lr 1e-3 --model-type custom --run-name custom-cnn-run
```

**Run 2 (MobileNetV3 Small):**
```bash
python3 train.py --epochs 5 --lr 5e-4 --model-type mobilenet --run-name mobilenet-run
```

### 3. Launch Streamlit UI
Run the Streamlit application:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. View MLflow UI
Launch the MLflow tracking interface:
```bash
mlflow ui --backend-store-uri artifacts/mlflow
```
Open [http://localhost:5000](http://localhost:5000) to view and compare your training runs.

---

## Docker Containerization

To package and run the Streamlit application inside a Docker container:

**1. Build the Docker Image:**
```bash
docker build -t bangla-ocr-app .
```

**2. Run the Docker Container:**
```bash
docker run -p 8501:8501 bangla-ocr-app
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the application.
