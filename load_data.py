import kagglehub

# Cargar path
#path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
#path = path + "/Data"
#print("Path to dataset files:", path)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


IMG_SIZE = (224, 224)


#funcion para cargar imagen
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalizar valores entre 0 y 1
    return img


def load_dataset(path):
  #Cargar path a carpetas
  healthy_path = os.path.join(path, "Normal")
  tumor_path = os.path.join(path, "Tumor")

  #Cargar path carpetas tumores
  glioma_path = os.path.join(tumor_path, "glioma_tumor")
  meningioma_path = os.path.join(tumor_path, "meningioma_tumor")
  pituitary_path = os.path.join(tumor_path, "pituitary_tumor")

  #Cargar imagenes
  healthy_images = sorted(os.listdir(healthy_path))
  glioma_images = sorted(os.listdir(glioma_path))
  meningioma_images = sorted(os.listdir(meningioma_path))
  pituitary_images = sorted(os.listdir(pituitary_path))

  x = [] #imagenes
  y = [] #etiquetas

  #Pasar las imagenes a arrays
  for img_name in healthy_images:
    img_path = os.path.join(healthy_path, img_name)
    x.append(load_and_preprocess_image(img_path))
    y.append(0)

  for img_name in glioma_images:
    img_path = os.path.join(glioma_path, img_name)
    x.append(load_and_preprocess_image(img_path))
    y.append(1)

  for img_name in meningioma_images:
    img_path = os.path.join(meningioma_path, img_name)
    x.append(load_and_preprocess_image(img_path))
    y.append(2)

  for img_name in pituitary_images:
    img_path = os.path.join(pituitary_path, img_name)
    x.append(load_and_preprocess_image(img_path))
    y.append(3)

  #convertir a array de numpy
  x = np.array(x)
  y = np.array(y)

  y = to_categorical(y, num_classes=4)
  # Separar en entrenamiento y prueba
  return train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)


# Si ejecutamos este archivo directamente, cargamos el dataset
if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
    dataset_path = dataset_path + "/Data"
    X_train, X_test, y_train, y_test = load_dataset(dataset_path)
    print(f"Datos cargados: {len(X_train)} entrenamiento, {len(X_test)} prueba")


#Intentar imprimir un cerebro sano
#img_path = os.path.join(healthy_path, healthy_images[0])
#img = cv2.imread(img_path)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV carga en BGR, lo convertimos a RGB

#plt.imshow(img)
#plt.axis("off")
#plt.savefig("output.png") # Guarda la imagen en un archivo
#plt.close()
#print("Imagen guardada como output.png")

#prueba con cerebros con tumor
#img_path2 = os.path.join(glioma_path, glioma_images[0])
#img2 = cv2.imread(img_path2)
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # OpenCV carga en BGR, lo convertimos a RGB

#plt.imshow(img2)
#plt.axis("off")
#plt.savefig("output2.png") # Guarda la imagen en un archivo
#plt.close()

#todo funciona bien 
