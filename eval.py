import torch
import torch.nn as nn
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------
# MODEL ARCHITECTURE (exactly what you trained)
# ----------------------------------------------------
class CustomEmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomEmotionCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 24×24

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Output: 12×12

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Output: 6×6

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # Classifier
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ----------------------------------------------------
# LOAD MODEL + WEIGHTS
# ----------------------------------------------------
num_classes = 9
model = CustomEmotionCNN(num_classes)
model.load_state_dict(torch.load("navarasa_emotion_model_custom_split2.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------------------------------
# CLASS LABELS
# ----------------------------------------------------
class_labels = [
    'ADBHUTA', 'BHAYANAKA', 'BIBHATSYA', 'HASYA', 'KARUNA',
    'RAUDRA', 'SHANTA', 'SHRINGARA', 'VEERA'
]

# ----------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------
def preprocess_face(img_path):
    img = cv2.imread(img_path)

    # ---- IMPORTANT FIX ----
    # Your model expects 3 channels (RGB), not grayscale.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(img, (48, 48))
    normalized = resized / 255.0

    # Convert to CHW and add batch dim => (1, 3, 48, 48)
    tensor_img = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return tensor_img.to(device)

# ----------------------------------------------------
# INFERENCE
# ----------------------------------------------------
face_img_path = "extracted_dataset/NAVRASA FACIAL EMOTION IMAGE DATA/training/SHANTA/S149_002_00000012.png"

input_img = preprocess_face(face_img_path)

with torch.no_grad():
    output = model(input_img)
    pred_idx = torch.argmax(output, dim=1).item()
    predicted_class = class_labels[pred_idx]

print(f"\nPredicted Emotion: {predicted_class}\n")
