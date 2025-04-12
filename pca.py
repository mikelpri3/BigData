from sklearn.decomposition import PCA
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt
from load_data import axial_dataset

label_map = {
    0: "glioma",
    1: "meningioma",
    2: "pituitary",
    3: "normal"
}

colors = {
    "glioma": "red",
    "meningioma": "blue",
    "pituitary": "green",
    "normal": "orange"
}


def pca_by_classes():
    #(C, H, W)
    img_shape = (3, 224, 224)

    for class_id, class_name in label_map.items():
        
        class_images = []
        for img, label in axial_dataset:
            if (label.item() if torch.is_tensor(label) else label) == class_id:
                img_np = img.numpy()
                img_resized = cv2.resize(img_np.transpose(1, 2, 0), (img_shape[2], img_shape[1]))  # (H, W, C)
                img_resized = img_resized.transpose(2, 0, 1)  #(C, H, W)
                class_images.append(img_resized.flatten())

        class_images = np.array(class_images)

        print(f"  {class_images.shape[0]} images")

        # PCA
        pca = PCA(n_components=3)
        pca.fit(class_images)


        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            component = pca.components_[i].reshape(img_shape)  # (C, H, W)
            component_img = np.transpose(component, (1, 2, 0))  # (H, W, C)


            component_img -= component_img.min()
            component_img /= component_img.max()

            axs[i].imshow(component_img)
            axs[i].axis('off')
            axs[i].set_title(f'{i+1}º component')

        plt.suptitle(f"PCA - Principal Component for class: {class_name}")
        plt.tight_layout()
        plt.show()


def full_pca():
    data = []
    labels = []

    for img, label in axial_dataset:
        img_np = img.numpy().transpose(1, 2, 0)  # from (C, H, W) to (H, W, C)
        resized = cv2.resize(img_np, (224, 224))
        data.append(resized.flatten())
        labels.append(label.item() if torch.is_tensor(label) else label)

    data_matrix = np.array(data)
    print(f"Data matrix shape: {data_matrix.shape}")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_matrix)

    # Visualización por clase
    plt.figure(figsize=(10, 8))
    for class_id, class_name in label_map.items():
        indices = [i for i, lbl in enumerate(labels) if lbl == class_id]
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1],
                    label=class_name, alpha=0.7, color=colors[class_name])

    plt.title("PCA de brain_dataset por clase")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
