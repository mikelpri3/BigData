from sklearn.manifold import TSNE
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

data = []
labels = []

for img, label in axial_dataset:
    img_np = img.numpy().transpose(1, 2, 0)  # (H, W, C)
    resized = cv2.resize(img_np, (224, 224))
    data.append(resized.flatten())
    labels.append(label.item() if torch.is_tensor(label) else label)

data_matrix = np.array(data)
print(f"Data matrix shape: {data_matrix.shape}")


tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(data_matrix)


plt.figure(figsize=(10, 8))
for class_id, class_name in label_map.items():
    indices = [i for i, lbl in enumerate(labels) if lbl == class_id]
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                label=class_name, alpha=0.7, color=colors[class_name])

plt.title("t-SNE")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.show()
