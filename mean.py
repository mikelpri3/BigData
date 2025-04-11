import numpy as np
import matplotlib.pyplot as plt
import cv2
from load_data import axial_dataset
IMG_SIZE = (224, 224)

images = []
labels = []

for img, label in axial_dataset:
    images.append(img.numpy())  # img: (C, H, W)
    labels.append(label)

# To arrays
images = np.array(images, dtype=np.float32)  # (N, C, H, W)
labels = np.array(labels)

# Mean image by class
mean_images_by_class = {}
class_names = axial_dataset.classes


for class_index, class_name in enumerate(class_names):
    class_images = images[labels == class_index]  # (Nc, C, H, W)
    class_mean = np.mean(class_images, axis=0)  # (C, H, W)
    class_mean_np = np.transpose(class_mean, (1, 2, 0)).astype(np.float32)  # (H, W, C)
    resized_mean = cv2.resize(class_mean_np, IMG_SIZE)

    mean_images_by_class[class_name] = resized_mean

    plt.figure(figsize=(5, 5))
    plt.imshow(resized_mean)
    plt.title(f"{class_name} average image")
    plt.axis("off")
    plt.show()


normal_mean = mean_images_by_class["Normal"]
pituitary_mean = mean_images_by_class["Tumor/pituitary_tumor"]
glioma_mean = mean_images_by_class["Tumor/glioma_tumor"]
meningioma_mean = mean_images_by_class["Tumor/meningioma_tumor"]

def mean_diff(normal_mean, tumor_mean, tumor):
    mean_diff = np.sqrt((normal_mean)**2 - (tumor_mean)**2)
    plt.figure(figsize=(5,5))
    plt.imshow(mean_diff)
    plt.title("diff " + f"{tumor}" + " image")
    plt.axis("off")
    plt.show()