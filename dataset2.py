import kagglehub
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


dataset2_path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
dataset2_path = os.path.join(dataset2_path, "Testing")

IMG_SIZE = (224, 224)

class BrainTumorDataset2(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
        self.images = []
        self.labels = []

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
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        label = self.labels[idx]
        return img, label


#Train-test obtainance by stratify

brain_dataset2 = BrainTumorDataset2(dataset2_path)

labels2 = brain_dataset2.labels

from plots import show_random_samples

show_random_samples(brain_dataset2,brain_dataset2.classes)