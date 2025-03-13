import kagglehub
from load_data import load_dataset
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert NumPy image to PyTorch tensor
])


# Create dataset
brain_dataset = BrainTumorDataset(dataset_path, transform=transform)

# Split dataset into training and testing
train_size = int(0.8 * len(brain_dataset))
test_size = len(brain_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(brain_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import matplotlib.pyplot as plt

# # Get one batch of images
data_iter = iter(train_loader)
images, labels = next(data_iter)

# # Convert tensor to NumPy image for display
image = images[0].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

# # Plot the image
plt.imshow(image)
plt.axis("off")
plt.savefig("output_model.png")
plt.close()
print("Imagen guardada como output_model.png")


######################  HASTA AQUI FUNCIONA   ####################################



