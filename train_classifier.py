import pandas as pd
import cv2
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Parameters
IMAGE_DIR = "images"  # Directory containing the images
CSV_PATH = "images/annotations.csv"  # CSV file with annotation data
ROI_SIZE = (32, 32)  # Fixed size for ROI extraction

# Read the CSV
df = pd.read_csv(CSV_PATH)

# Initialize lists for features and labels
features = []
labels = []

for index, row in df.iterrows():
    filename = row['filename']
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load {img_path}")
        continue

    # Extract ROI using the provided bounding box
    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    roi = img[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        continue

    # Resize ROI and flatten to a feature vector
    roi_resized = cv2.resize(roi, ROI_SIZE)
    feat = roi_resized.flatten()
    features.append(feat)
    labels.append(row['class'])

features = np.array(features)
labels = np.array(labels)

# Split data (for evaluation)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a classifier (you can experiment with other models too)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(clf, "classifier.pkl")
print("Trained model saved as 'classifier.pkl'")
