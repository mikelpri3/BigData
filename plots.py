import kagglehub
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
dataset_path = os.path.join(dataset_path, "Data")


#Load Dataset
IMG_SIZE = (224, 224)

from model import BrainTumorDataset, set_seed
set_seed(42)

# Create dataset
brain_dataset = BrainTumorDataset(dataset_path)

# PLOT 1: Conteo de imágenes por clase
def count_of_classes(brain_dataset):
    counts = [brain_dataset.labels.count(i) for i in range(len(brain_dataset.classes))]
    plt.bar(brain_dataset.classes, counts, color='skyblue')
    plt.xticks(rotation=0)
    plt.title("Images Distribution per Class")
    plt.ylabel("Amount of images")
    plt.savefig("amount_classes.png") # Guarda la imagen en un archivo
    plt.close()
    print("Imagen guardada como amount_classes.png")

# PLOT 2: Random Samples
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
    plt.savefig("random_samples.png") # Guarda la imagen en un archivo
    plt.close()
    print("Imagen guardada como random_samples.png")

###################################  Re-Load Model  #########################################3

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# Split dataset into training and testing
train_size = int(0.8 * len(brain_dataset))
test_size = len(brain_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(brain_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

from model import CNNModel

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
from model import train_model

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=4)


#PLOT3: Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion_Matrix")
    plt.savefig("Confusion_Matrix.png") # Guarda la imagen en un archivo
    plt.close()
    print("Imagen guardada como Confusion_Matrix.png")

#PLOT 4: CURVA ROC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import itertools

def plot_roc_curves(model, test_loader, classes):
    model.eval()
    y_true, y_score = [], []
    num_classes = len(classes)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_score.extend(probabilities)
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    colors = itertools.cycle(["blue", "red", "green", "purple", "orange", "brown", "pink", "gray", "cyan", "magenta"])
    
    for i, color in zip(range(num_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{classes[i]} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 0.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by classes")
    plt.legend(loc="lower right")
    plt.savefig("ROC_Curves.png")  # Guarda la imagen en un archivo
    plt.close()
    print("Imagen guardada como ROC_Curves.png")

#Ejecucion de los plots
count_of_classes(brain_dataset)
show_random_samples(brain_dataset, brain_dataset.classes)
plot_confusion_matrix(model, test_loader, brain_dataset.classes)
plot_roc_curves(model, test_loader, brain_dataset.classes)
