import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")

print("Path to dataset files:", path)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

image_files = []
labels = []

for category in os.listdir(path):
    category_path = os.path.join(path, category)
    if os.path.isdir(category_path):  # Ensure it's a directory
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            image = cv2.imread(img_path)  # Read image
            image = cv2.resize(image, (224, 224))  # Resize image if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image_files.append(image)
            labels.append(category)

image_files = np.array(image_files)
labels = np.array(labels)

# Display one sample image
plt.imshow(image_files[0])
plt.title(labels[0])
plt.axis("off")
plt.show()