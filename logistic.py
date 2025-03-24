import kagglehub
import os
from load_data import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

dataset_path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
dataset_path = os.path.join(dataset_path, "Data")

X_train, X_test, y_train, y_test = load_dataset(dataset_path)

# Flatten the images for Logistic Regression
X_train = X_train.reshape(X_train.shape[0], -1)  # [num_samples, num_features]
X_test = X_test.reshape(X_test.shape[0], -1)  # [num_samples, num_features]

# Initialize Logistic Regression for Multiclass Classification
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
