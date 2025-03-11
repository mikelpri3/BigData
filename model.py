import tensorflow as tf
import kagglehub
import os 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from load_data import load_dataset, path  # Importamos la función para cargar los datos

# Definir la función para construir el modelo
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Si ejecutamos este archivo directamente, entrenamos el modelo
if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
    dataset_path = os.path.join(dataset_path, "Data")
    X_train, X_test, y_train, y_test = load_dataset(dataset_path)

    model = create_model()
    model.summary()

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Precisión en datos de prueba: {test_acc:.2%}")

    model.save("modelo_tumores.h5")
