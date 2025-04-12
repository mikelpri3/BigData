import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from load_data import axial_dataset

IMG_SIZE = (224, 224)

def load_images(axial_dataset):
    images = []
    labels = []

    for img, label in axial_dataset:
        images.append(img.numpy())  # img: (C, H, W)
        labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels)

def mean_image(images, labels):
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
        
    return mean_images_by_class

def mean_diff(normal_mean, tumor_mean, tumor):
    mean_diff = np.sqrt((normal_mean)**2 - (tumor_mean)**2)
    plt.figure(figsize=(5,5))
    plt.imshow(mean_diff)
    plt.title("diff " + f"{tumor}" + " image")
    plt.axis("off")
    plt.show()


def pca_analysis(n_components=50):
    images, labels = load_images(axial_dataset)
    # from (N, C, H, W) to (N, C*H*W)
    flattened_images = images.reshape(images.shape[0], -1)  # (N, C*H*W)

    # Compute PCA
    n_components = 50
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(flattened_images)
    return pca, X_pca, images


def normalized_components(pca, images):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        component = pca.components_[i].reshape(images.shape[1], images.shape[2], images.shape[3])  # (C, H, W)
        component_img = np.transpose(component, (1, 2, 0))  # (H, W, C)
        
        component_img -= component_img.min()
        component_img /= component_img.max()

        axs[i].imshow(component_img)
        axs[i].axis('off')
        axs[i].set_title(f'{i+1}º component')

    plt.suptitle("First PCA components (Normalized)")
    plt.show()

if __name__ == "__main__":
    images, labels = load_images(axial_dataset)
    mean_images_by_class = mean_image(images, labels)
    normal_mean = mean_images_by_class["Normal"]
    pituitary_mean = mean_images_by_class["Tumor/pituitary_tumor"]
    glioma_mean = mean_images_by_class["Tumor/glioma_tumor"]
    meningioma_mean = mean_images_by_class["Tumor/meningioma_tumor"]
