from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt
from mean import images

# from (N, C, H, W) to (N, C*H*W)
flattened_images = images.reshape(images.shape[0], -1)  # (N, C*H*W)

# Compute PCA
n_components = 50
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(flattened_images)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

def normalized_components():
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