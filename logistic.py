import kagglehub

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from load_data import load_dataset, load_and_preprocess_image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
dataset_path = os.path.join(dataset_path, "Data")
x, y = load_dataset(dataset_path)
X_train, X_test, y_train, y_test = load_dataset(dataset_path)

model = LogisticRegression()
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))