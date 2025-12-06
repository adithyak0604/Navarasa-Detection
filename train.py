import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import copy

# ==========================================
# 1. CONFIGURATION
# ==========================================
# UPDATE THESE PATHS to match your folder structure exactly
base_path = 'extracted_dataset/NAVRASA FACIAL EMOTION IMAGE DATA'
train_dir = os.path.join(base_path, 'training') 
test_dir = os.path.join(base_path, 'testing')   

IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 16
EPOCHS = 25
NUM_CLASSES = 9 
LEARNING_RATE = 0.001

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA TRANSFORMS & LOADING
# ==========================================

# A. Training Transforms (Augmentation)
# Matches Keras: rotation, shifts (translate), flip, zoom (scale)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(), # Converts 0-255 to 0.0-1.0 automatically
])

# B. Testing/Validation Transforms (No Augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

print("Loading Data...")
try:
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Classes found: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please check if the paths exist and contain images in subfolders.")
    exit()

# ==========================================
# 3. DEFINE CUSTOM CNN MODEL
# ==========================================
class CustomEmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomEmotionCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 24x24
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Pool -> Output: 12x12
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Pool -> Output: 6x6
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 128) # 6x6 spatial dimension comes from 48/(2*2*2)
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
        x = self.fc2(x) # Returns raw logits (no softmax here, CrossEntropyLoss handles it)
        return x

model = CustomEmotionCNN(NUM_CLASSES).to(device)
print(model)

# ==========================================
# 4. TRAINING SETUP
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early Stopping parameters
patience = 5
best_val_loss = float('inf')
patience_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

# History for plotting
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("\nStarting Training...")

for epoch in range(EPOCHS):
    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()           # Clear gradients
        outputs = model(inputs)         # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = correct_train / total_train
    
    # --- VALIDATION PHASE ---
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad(): # No gradient needed for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = correct_val / total_val
    
    # Store history
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
    
    # --- EARLY STOPPING CHECK ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0 # Reset counter
        # Save the best model
        torch.save(model.state_dict(), 'navarasa_emotion_model_custom_split2.pth')
    else:
        patience_counter += 1
        print(f"EarlyStopping counter: {patience_counter} out of {patience}")
        if patience_counter >= patience:
            print("Early Stopping triggered.")
            break

# Load best weights before finishing
model.load_state_dict(best_model_wts)
print("Training Complete. Best model loaded.")

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Final Evaluation
model.eval()
final_loss = 0.0
final_correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        final_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        final_correct += (predicted == labels).sum().item()

print(f"\nFinal Test Loss: {final_loss/total:.4f}")
print(f"Final Test Accuracy: {final_correct/total:.4f}")
