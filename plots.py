import kagglehub
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# PLOT 1: Conteo de imágenes por clase
def count_of_classes(brain_dataset):
    counts = [brain_dataset.labels.count(i) for i in range(len(brain_dataset.classes))]
    plt.bar(brain_dataset.classes, counts, color='skyblue')
    plt.xticks(rotation=20, ha="right")
    plt.title("Images Distribution per Class")
    plt.ylabel("Amount of images")
    plt.tight_layout()
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
    plt.show()


# PLOT 3: Training Loss
def plot_training_loss(history):
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(history['loss']) + 1), history['loss'], marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid()
    plt.show()



# PLOT 4: Confusion Matrix
from cnn import device
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
    plt.show()


#Plot 5: ROC Curve
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
    plt.show()