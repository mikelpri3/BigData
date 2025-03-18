import kagglehub
from load_data import load_dataset
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
dataset_path = os.path.join(dataset_path, "Data")



IMG_SIZE = (224, 224)

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["Normal", "Tumor/glioma_tumor", "Tumor/meningioma_tumor", "Tumor/pituitary_tumor"]
        self.images = []
        self.labels = []

        # Load all image paths
        for label, category in enumerate(self.classes):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, IMG_SIZE)  # Resize
        img = img / 255.0  # Normalize
        img = np.transpose(img, (2, 0, 1))  # Rearrange dimensions to (C, H, W)
        img = torch.tensor(img, dtype=torch.float32)  # Convert to tensor
        # if self.transform:
        #     img = self.transform(img)

        label = self.labels[idx]
        return img, label


transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((224, 224)),  # Resize if necessary
    transforms.ToTensor(),  # Convert NumPy image to PyTorch tensor
])


# Create dataset
brain_dataset = BrainTumorDataset(dataset_path, transform=transform)

# Conteo de imágenes por clase
counts = [brain_dataset.labels.count(i) for i in range(len(brain_dataset.classes))]
plt.bar(brain_dataset.classes, counts, color='skyblue')
plt.xticks(rotation=45)
plt.title("Distribución de imágenes por clase")
plt.ylabel("Cantidad de imágenes")

plt.savefig("amount_classes.png") # Guarda la imagen en un archivo
plt.close()
print("Imagen guardada como amount_classes.png")

def show_random_samples(dataset, classes):
    fig, axes = plt.subplots(len(classes), 5, figsize=(15, 10))
    for class_idx, class_name in enumerate(classes):
        class_indices = [i for i, label in enumerate(dataset.labels) if label == class_idx]
        random_samples = np.random.choice(class_indices, size=5, replace=False)
        for j, idx in enumerate(random_samples):
            img, _ = dataset[idx]
            img = np.transpose(img.numpy(), (1, 2, 0))
            axes[class_idx, j].imshow(img)
            axes[class_idx, j].axis('off')
            if j == 0:
                axes[class_idx, j].set_title(class_name)
    plt.show() 
    plt.savefig("random_samples.png") # Guarda la imagen en un archivo
    plt.close()
    print("Imagen guardada como random_samples.png")

show_random_samples(brain_dataset, brain_dataset.classes)

from sklearn.metrics import confusion_matrix
import seaborn as sns







from sklearn.metrics import confusion_matrix
import seaborn as sns
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 4)  # 4 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# Split dataset into training and testing
train_size = int(0.8 * len(brain_dataset))
test_size = len(brain_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(brain_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=4)

def plot_confusion_matrix(model, test_loader, classes):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Matriz de confusión")
    plt.savefig("Confusion_Matrix.png") # Guarda la imagen en un archivo
    plt.close()
    print("Imagen guardada como Confusion_Matrix.png")


plot_confusion_matrix(model, test_loader, brain_dataset.classes)
