#!/usr/bin/env python3
import json
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Define the CNN architecture (must match train.py)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=60):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Model architecture matching MobileNet variant
class MobileNetWrapper(nn.Module):
    def __init__(self, num_classes=60):
        super().__init__()
        # Import models inside to avoid errors if torchvision is not fully ready
        from torchvision import models
        self.model = models.mobilenet_v3_small(weights=None)
        
        # Modify first conv to accept 1 channel (grayscale)
        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load labels and model
@st.cache_resource
def load_resources(model_path):
    project_dir = Path(__file__).resolve().parent
    labels_file = project_dir / "labels.json"
    
    # Load labels
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_map = json.load(f)
        
    # Load model
    try:
        # The model saved by torch.save can be loaded directly
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        # Failback to architecture initialization and state_dict loading if required
        st.warning(f"Direct load failed, falling back to manual model initialization. Error: {e}")
        # Try custom CNN first
        model = CustomCNN(num_classes=60)
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict)
            else:
                model = state_dict
        except Exception as e2:
            st.error(f"Failed to load model weights: {e2}")
            return None, None
            
    model.eval()
    return model, labels_map

def merge_bounding_boxes(boxes, threshold_x=25):
    """
    Merge bounding boxes that are horizontally close to each other.
    This handles multi-stroke characters and horizontal modifiers.
    """
    if not boxes:
        return []
    
    # Sort boxes primarily by their X coordinate (left to right)
    boxes = sorted(boxes, key=lambda b: b[0])
    
    merged = []
    # Convert first box [x, y, w, h] to [x1, y1, x2, y2]
    curr = [boxes[0][0], boxes[0][1], boxes[0][0] + boxes[0][2], boxes[0][1] + boxes[0][3]]
    
    for box in boxes[1:]:
        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # If the horizontal start of the next box is within threshold of the current box's end
        if x1 <= curr[2] + threshold_x:
            # Merge intervals (expand horizontal and vertical boundaries)
            curr[0] = min(curr[0], x1)
            curr[1] = min(curr[1], y1)
            curr[2] = max(curr[2], x2)
            curr[3] = max(curr[3], y2)
        else:
            # Save the current box and start a new one
            merged.append(curr)
            curr = [x1, y1, x2, y2]
            
    merged.append(curr)
    
    # Convert back to x, y, w, h
    return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in merged]

def preprocess_and_segment(image_array, min_box_size=8, padding_ratio=0.15):
    """
    Preprocess drawn image and segment it into characters using contours.
    """
    # Convert RGBA to grayscale. Alpha channel carries drawing info.
    # The canvas uses white strokes on a black background
    gray = image_array[:, :, 3]
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out tiny noise boxes
        if w >= min_box_size and h >= min_box_size:
            raw_boxes.append((x, y, w, h))
            
    # Merge horizontally overlapping/close strokes
    merged_boxes = merge_bounding_boxes(raw_boxes)
    
    segmented_images = []
    for x, y, w, h in merged_boxes:
        # Crop the segment
        crop = binary[y:y+h, x:x+w]
        
        # Pad the cropped image to make it square without stretching
        size = max(w, h)
        padded = np.zeros((size, size), dtype=np.uint8)
        
        # Paste the cropped image in the center
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = crop
        
        # Add extra padding around the character to avoid edge artifacts
        extra_pad = int(size * padding_ratio)
        if extra_pad > 0:
            padded = cv2.copyMakeBorder(padded, extra_pad, extra_pad, extra_pad, extra_pad, cv2.BORDER_CONSTANT, value=0)
            
        # Resize to model input size (64x64)
        resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_LINEAR)
        segmented_images.append(resized)
        
    return segmented_images, merged_boxes, binary

def predict_character(model, img_64x64, labels_map):
    # Normalize image to match PyTorch tensor transforms: (pixel / 255.0 - 0.5) / 0.5
    img_normalized = (img_64x64.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 64, 64)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probs, 3)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        class_folder_name = str(idx.item() + 1)
        char = labels_map.get(class_folder_name, f"Unknown ({class_folder_name})")
        results.append((char, prob.item()))
        
    return results

def main():
    st.set_page_config(page_title="Bangla OCR Word Recognizer", page_icon="✍️", layout="wide")
    
    # Custom CSS for modern design aesthetics
    st.markdown("""
        <style>
            .main { background-color: #0f1116; color: #e2e8f0; }
            h1 { color: #38bdf8; font-weight: 800; font-family: 'Inter', sans-serif; }
            .recognized-word { font-size: 64px; font-weight: bold; color: #10b981; text-align: center; margin: 20px 0; border: 2px dashed #10b981; padding: 10px; border-radius: 10px; background-color: #064e3b; }
            .stButton>button { background-color: #38bdf8; color: #0f1116; font-weight: bold; border-radius: 8px; border: none; padding: 10px 24px; transition: all 0.3s; }
            .stButton>button:hover { background-color: #0ea5e9; transform: scale(1.05); }
        </style>
    """, unsafe_allow_html=True)

    st.title("Bangla Handwritten Word Recognizer")
    st.markdown("Draw a single Bangla word (using basic letters/digits, e.g. **১২**, **আম**, **কলম**) on the canvas below. The application will segment the word into characters and predict them using a trained CNN.")
    
    # Sidebar config
    with st.sidebar:
        st.header("Configuration")
        model_path_input = st.text_input("Model File Path", value="models/model.pkl")
        
        # st.markdown("---")
        # st.subheader("💡 Character Set Info")
        # st.write("This model classifies the **60 basic Bangla characters**:")
        # st.write("- **11 Vowels** (অ to ঔ)")
        # st.write("- **39 Consonants & Diacritics** (ক to ঁ)")
        # st.write("- **10 Numerals** (০ to ৯)")
        # st.write("*Note: Compound words/conjuncts are excluded.*")
        
    # Ensure model exists
    model_path = Path(model_path_input)
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}. Please train a model first using `train.py`.")
        st.stop()
        
    # Load model and mapping
    model, labels_map = load_resources(model_path)
    if model is None:
        st.stop()
        
    # Main columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Drawing Canvas")
        st.info("Tip: Draw characters separated horizontally (left-to-right) for best segmentation results.")
        
        # Create a drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=12,
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=220,
            width=700,
            drawing_mode="freedraw",
            key="canvas",
        )
        
    # If the user has drawn on the canvas
    if canvas_result.image_data is not None:
        # Check if the canvas has non-zero drawing pixels
        has_drawings = np.any(canvas_result.image_data[:, :, 3] > 0)
        
        if has_drawings:
            # Perform segmentation and preprocessing
            char_images, bounding_boxes, binary_img = preprocess_and_segment(canvas_result.image_data)
            
            with col2:
                st.subheader("Segmentation & Binary View")
                # Scale binary image down for display
                resized_binary = cv2.resize(binary_img, (350, 110))
                st.image(resized_binary, caption="Binarized Canvas Image", clamp=True)
                st.success(f"Detected {len(char_images)} character strokes/regions.")
                
            if char_images:
                # Perform predictions
                predicted_chars = []
                confidences = []
                top_3_predictions = []
                
                for img in char_images:
                    preds = predict_character(model, img, labels_map)
                    predicted_chars.append(preds[0][0])
                    confidences.append(preds[0][1])
                    top_3_predictions.append(preds)
                    
                # Display final recognized word
                recognized_word = "".join(predicted_chars)
                st.markdown("---")
                st.subheader("Word Recognition Result")
                st.markdown(f'<div class="recognized-word">{recognized_word}</div>', unsafe_allow_html=True)
                
                # Show per-character predictions
                st.subheader("Detailed Character Predictions")
                cols = st.columns(min(len(char_images), 6))
                
                for idx, (img, top_preds) in enumerate(zip(char_images, top_3_predictions)):
                    col_idx = idx % 6
                    with cols[col_idx]:
                        # Show cropped character image
                        st.image(img, caption=f"Char {idx+1}", width=90)
                #         # Show top predictions
                        for rank, (char, score) in enumerate(top_preds):
                            if rank == 0:
                                st.markdown(f"**{char}**: `{score*100:.1f}%` 🥇")
                            elif rank == 1:
                                st.markdown(f"{char}: `{score*100:.1f}%` 🥈")
                            else:
                                st.markdown(f"{char}: `{score*100:.1f}%` 🥉")
        else:
            st.warning("Draw something on the canvas to trigger character recognition!")

if __name__ == "__main__":
    main()

#