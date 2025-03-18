import kagglehub
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
  healthy_images = os.listdir(healthy_path)
  glioma_images = os.listdir(glioma_path)
  meningioma_images = os.listdir(meningioma_path)
  pituitary_images = os.listdir(pituitary_path)

  #max_images = 500  # Ajusta según la memoria disponible para ejecutarlo en el portatil
  #healthy_images = healthy_images[:max_images]
  #glioma_images = glioma_images[:max_images]
  #meningioma_images = meningioma_images[:max_images]
  #pituitary_images = pituitary_images[:max_images]

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

  # Separar en entrenamiento y prueba
  return train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)


# Si ejecutamos este archivo directamente, cargamos el dataset (codigo para hacer el main)
if __name__ == "__main__":
  dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
  dataset_path = os.path.join(dataset_path, "Data")
  X_train, X_test, y_train, y_test = load_dataset(dataset_path)
  print(f"Datos cargados: {len(X_train)} entrenamiento, {len(X_test)} prueba")


