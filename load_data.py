import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")

print("Path to dataset files:", path)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#Cargar path a dataset
path = path + "/Data"

#Cargar path a carpetas
healthy_path = os.path.join(path, "Normal")
tumor_path = os.path.join(path, "Tumor")

#Cargar path carpetas tumores
glioma_path = os.path.join(tumor_path, "glioma_tumor")
meningioma_path = os.path.join(tumor_path, "meningioma_tumor")
pituitary_path = os.path.join(tumor_path, "pituitary_tumor")

#Cargar imagenes de las carpetas
healthy_images = os.listdir(healthy_path)
glioma_images = os.listdir(glioma_path)
meningioma_images = os.listdir(meningioma_path)
pituitary_images = os.listdir(pituitary_path)


#Intentar imprimir un cerebro sano
img_path = os.path.join(healthy_path, healthy_images[0])
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV carga en BGR, lo convertimos a RGB

plt.imshow(img)
plt.axis("off")
plt.savefig("output.png") # Guarda la imagen en un archivo
plt.show("output.png") 
plt.close()
print("Imagen guardada como output.png")




















############################################################################################################
#image_files = []
#labels = []

#for category in os.listdir(path):
  #  category_path = os.path.join(path, category)
  #  if os.path.isdir(category_path):  # Ensure it's a directory
   #     for img_name in os.listdir(category_path):
  #          img_path = os.path.join(category_path, img_name)
   #         image = cv2.imread(img_path)  # Read image
   #         image = cv2.resize(image, (224, 224))  # Resize image if needed
    #        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    #        image_files.append(image)
    #        labels.append(category)

#image_files = np.array(image_files)
#labels = np.array(labels)

# Display one sample image
#plt.imshow(image_files[0])
#plt.title(labels[0])
#plt.axis("off")
#plt.show()