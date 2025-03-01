import kagglehub
import numpy as np
import tensorflow as tf
from load_data import load_and_preprocess_image, path
import os

# Cargar modelo entrenado
model = tf.keras.models.load_model("modelo_tumores.h5")

# Función para predecir con una imagen nueva
def predict_image(img_path):
    img = load_and_preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para la predicción
    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        print("Predicción: Cerebro con tumor")
    else:
        print("Predicción: Cerebro sano")

# Prueba con nuevas imágenes
dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
dataset_path = dataset_path + "/Data"
healthy_path = os.path.join(dataset_path, "Normal")
tumor_path = os.path.join(dataset_path, "Tumor/glioma_tumor")

healthy_images = os.listdir(healthy_path)
tumor_images = os.listdir(tumor_path)

# Probar con imágenes reales
predict_image(os.path.join(healthy_path, healthy_images[0]))  # Imagen sana
predict_image(os.path.join(tumor_path, tumor_images[0]))  # Imagen con tumor
