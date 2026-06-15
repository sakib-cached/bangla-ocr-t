#!/usr/bin/env python3
import os
import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define dataset class that loads images from folders 1 to 60
class BanglaLekhaDataset(Dataset):
    def __init__(self, base_dir, transform=None, num_classes=60):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.num_classes = num_classes
        
        self.samples = []
        # Traverse folder 1 to num_classes (60)
        for class_id in range(1, num_classes + 1):
            class_folder = self.base_dir / str(class_id)
            if not class_folder.exists():
                continue
            
            # Find all images in this folder
            for f in class_folder.iterdir():
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Label is 0-indexed: folder '1' is label 0, '2' is label 1, etc.
                    self.samples.append((str(f), class_id - 1))
                    
        print(f"Loaded {len(self.samples)} image paths from {num_classes} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Open image in grayscale (L mode)
            img = Image.open(img_path).convert('L')
        except Exception as e:
            # Fail-safe: create a black image if reading fails
            img = Image.new('L', (64, 64), 0)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

# Custom CNN Architecture for 64x64 Grayscale Images
class CustomCNN(nn.Module):
    def __init__(self, num_classes=60):
        super().__init__()
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32x32
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 16x16
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # Output: 8x8
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

# Function to build transfer learning model (MobileNetV3)
def build_mobilenet(num_classes=60, pretrained=True):
    # Load mobile net
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    
    # The original MobileNet takes 3 channels. We modify the first conv layer
    # to accept 1 channel (grayscale) while keeping pretrained weights for other layers
    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    # Average the weights across channels to initialize the new conv layer
    with torch.no_grad():
        new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_conv
    
    # Modify classifier head
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

# Train & Validation functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total

@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        
    return np.array(all_labels), np.array(all_preds)

# Plotting helpers
def plot_curves(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Acc', color='blue')
    axes[1].plot(history['val_acc'], label='Val Acc', color='red')
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, output_path, labels_map):
    # Select a subset of labels if classes are too many to avoid crowded plot
    # We display a subset of classes (e.g. first 20 classes)
    indices = np.where(y_true < 20)[0]
    cm = confusion_matrix(y_true[indices], y_pred[indices], labels=list(range(20)))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    display_labels = [labels_map[str(i+1)] for i in range(20)]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title("Confusion Matrix (Classes 1-20)")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Train a CNN on BanglaLekha-Isolated dataset.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model-type", type=str, default="custom", choices=["custom", "mobilenet"], help="Model architecture")
    parser.add_argument("--train-limit", type=int, default=15000, help="Limit number of train images for quick runs")
    parser.add_argument("--val-limit", type=int, default=3000, help="Limit number of val images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-name", type=str, default="bangla-ocr-training", help="Name of the MLflow run")
    parser.add_argument("--experiment-name", type=str, default="Bangla-OCR-Experiment", help="Name of the MLflow experiment")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    import mlflow
    import mlflow.pytorch
    
    mlflow_tracking_dir = Path("artifacts/mlflow").resolve()
    mlflow_tracking_dir.mkdir(parents=True, exist_ok=True)
    db_path = mlflow_tracking_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(args.experiment_name)
    
    # Paths
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "BanglaLekha-Isolated" / "Images"
    labels_file = project_dir / "labels.json"
    
    # Load labels mapping for plotting names
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels_map = json.load(f)
        
    # Set up Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define Data Transforms
    # Images in the dataset have varying sizes. We resize them to 64x64.
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Initialize Datasets
    full_train_dataset = BanglaLekhaDataset(dataset_dir, transform=train_transforms)
    full_val_dataset = BanglaLekhaDataset(dataset_dir, transform=val_transforms)
    
    # Split into train & validation indices deterministically
    num_samples = len(full_train_dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    # 85-15 Split
    split_idx = int(0.85 * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Apply Limits if specified (to speed up classroom trainings)
    if args.train_limit and args.train_limit < len(train_indices):
        train_indices = train_indices[:args.train_limit]
    if args.val_limit and args.val_limit < len(val_indices):
        val_indices = val_indices[:args.val_limit]
        
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Training subset size: {len(train_dataset)}, Validation subset size: {len(val_dataset)}")
    
    # Build Model
    if args.model_type == "custom":
        model = CustomCNN(num_classes=60)
        arch_description = "Custom CNN: Conv(32)-Conv(64)-Conv(128)-FC(256)-FC(60)"
    else:
        model = build_mobilenet(num_classes=60, pretrained=True)
        arch_description = "MobileNetV3 Small transfer learning with 1-channel grayscale input head"
        
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name) as run:
        # Log Hyperparameters & Metadata
        mlflow.log_params({
            "model_type": args.model_type,
            "architecture": arch_description,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "seed": args.seed,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "num_classes": 60,
            "device": str(device)
        })
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_model_weights = None
        
        # Training loop
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch + 1)
            
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Keep track of best model weights
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = model.state_dict()
                
        # Load best weights to compute final metrics
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
            
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        
        # Save model locally in models/model.pkl
        models_dir = project_dir / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "model.pkl"
        torch.save(model, model_path)
        print(f"Saved best model to {model_path}")
        
        # Log predictions and confusion matrix
        y_true, y_pred = get_predictions(model, val_loader, device)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            curve_file = tmp_path / "training_curves.png"
            cm_file = tmp_path / "confusion_matrix.png"
            
            # Plot and save curves and matrix
            plot_curves(history, curve_file)
            plot_confusion_matrix(y_true, y_pred, cm_file, labels_map)
            
            # Log artifacts to MLflow
            mlflow.log_artifact(str(curve_file), artifact_path="plots")
            mlflow.log_artifact(str(cm_file), artifact_path="plots")
            mlflow.log_artifact(str(labels_file), artifact_path="metadata")
            
        # Log model artifact to MLflow registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"Bangla-OCR-{args.model_type}"
        )
        
        print(f"Successfully completed run: {run.info.run_name}")

if __name__ == "__main__":
    main()
