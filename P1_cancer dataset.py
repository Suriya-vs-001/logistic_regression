# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data = pd.read_csv(url, header=None)

# Check the number of columns
print(f"The dataset has {data.shape[1]} columns.")
columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
print(columns)

# Step 2: Data Preprocessing
# Convert Diagnosis (B for benign, M for malignant) to binary (0 and 1)
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Drop ID column as it is not useful for the model
data.drop('ID', axis=1, inplace=True)

# Splitting data into features (X) and target (y)
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Build Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"F1-Score: {f1:.2f}")


# Step 5: K-Fold Cross-Validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracies = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"K-Fold Cross-Validation Mean Accuracy: {cv_accuracies.mean() * 100:.2f}%")
print(f"Standard Deviation of Accuracy: {cv_accuracies.std() * 100:.2f}%")
