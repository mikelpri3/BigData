import kagglehub
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
dataset_path = os.path.join(dataset_path, "Data")

# Load Dataset
IMG_SIZE = (224, 224)

from model import BrainTumorDataset, CNNModel

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

brain_dataset = BrainTumorDataset(dataset_path, transform=transform)

def plot_pca(dataset, labels, title, filename):
    images = [img.numpy().flatten() for img, _ in dataset]
    images = np.array(images)
    
    # Estandarizar los datos
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(images_scaled)
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=range(4), label='Clases')
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Imagen guardada como {filename}")

def generate_pca_plots(dataset, model, device):
    true_labels = np.array([label for _, label in dataset])
    predicted_labels = []
    
    model.eval()
    with torch.no_grad():
        for img, _ in dataset:
            img = img.unsqueeze(0).to(device)
            output = model(img)
            _, pred = torch.max(output, 1)
            predicted_labels.append(pred.item())
    
    plot_pca(dataset, true_labels, "PCA basado en etiquetas reales", "pca_true.png")
    plot_pca(dataset, np.array(predicted_labels), "PCA basado en etiquetas predichas", "pca_pred.png")

# Cargar modelo
model = CNNModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Generar los PCAs
generate_pca_plots(brain_dataset, model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
