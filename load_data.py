import kagglehub
import os
import cv2
import numpy as np
import torch 
from torch.utils.data import Dataset

dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
dataset_path = os.path.join(dataset_path, "Data") #Enter in the folder Data


IMG_SIZE = (224, 224)



#Function to differenciate images with axial perspective
def is_axial_based_on_black_border(img_path, border_thickness=10, black_thresh=10, black_ratio_thresh=0.99):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img = cv2.resize(img, (224, 224))

    # Extract borders
    top = img[:border_thickness, :]
    bottom = img[-border_thickness:, :]
    left = img[:, :border_thickness]
    right = img[:, -border_thickness:]

    # Unify all borders
    border_pixels = np.concatenate((top.flatten(), bottom.flatten(), left.flatten(), right.flatten()))
    
    # Compute proportion of black pixels
    black_pixels = np.sum(border_pixels < black_thresh)
    black_ratio = black_pixels / len(border_pixels)

    return black_ratio > black_ratio_thresh

###########################

#Dataset declaration
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None, view_type="axial"):
        self.root_dir = root_dir
        self.transform = transform
        self.view_type = view_type
        self.classes = ["Normal", "Tumor/glioma_tumor", "Tumor/meningioma_tumor", "Tumor/pituitary_tumor"]
        self.images = []
        self.labels = []

        for label, category in enumerate(self.classes):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if self.view_type == "all":
                    self.images.append(img_path)
                    self.labels.append(label)
                     
                if is_axial_based_on_black_border(img_path) and self.view_type == "axial":
                                    self.images.append(img_path)
                                    self.labels.append(label)
                elif not is_axial_based_on_black_border(img_path) and self.view_type == "sagittal":
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
    
brain_dataset = BrainTumorDataset(dataset_path, view_type="all") #load all images

axial_dataset = BrainTumorDataset(dataset_path, view_type="axial") #load axial images
