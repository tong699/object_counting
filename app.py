import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---------------------------
# Helper Functions for Training
# ---------------------------
def extract_features(roi, size=(32, 32)):
    """
    Resize ROI to fixed size and flatten it as a feature vector.
    """
    roi_resized = cv2.resize(roi, size)
    return roi_resized.flatten()

def train_classifier(csv_data, images_folder, roi_size=(32, 32)):
    """
    Reads CSV annotations and trains a Logistic Regression classifier.
    Expects the CSV to have columns: filename,width,height,class,xmin,ymin,xmax,ymax.
    Returns the classifier and a text report.
    """
    # Read the CSV data into a DataFrame
    df = pd.read_csv(csv_data)
    
    features = []
    labels = []
    
    missing_files = []
    for idx, row in df.iterrows():
        filename = row['filename']
        img_path = os.path.join(images_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            missing_files.append(filename)
            continue

        # Extract ROI using the bounding box from CSV
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        roi = img[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            continue
        
        feat = extract_features(roi, size=roi_size)
        features.append(feat)
        labels.append(row['class'])
    
    if len(features) == 0:
        return None, "No valid training data found."
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train the classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate the classifier
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return clf, report

def save_classifier(model, filename="classifier.pkl"):
    joblib.dump(model, filename)

@st.cache_resource(show_spinner=False)
def load_classifier(model_path="classifier.pkl"):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error("Classifier not found. Please train a classifier first.")
        return None

# ---------------------------
# Helper Functions for Demo Processing
# ---------------------------
def preprocess_image(image):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(gray_img, ksize):
    # Ensure ksize is odd and at least 1
    ksize = max(1, ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(gray_img, (ksize, ksize), 0)

def apply_threshold(blurred, method, global_thresh, block_size, C):
    if method == "Global":
        _, thresh = cv2.threshold(blurred, global_thresh, 255, cv2.THRESH_BINARY_INV)
    elif method == "Otsu":
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "Adaptive Gaussian":
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C)
    else:  # Adaptive Mean
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C)
    return thresh

def apply_morphology(thresh, method, morph_ksize):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
    if method == "None":
        return thresh
    elif method == "Erosion":
        return cv2.erode(thresh, kernel, iterations=1)
    elif method == "Dilation":
        return cv2.dilate(thresh, kernel, iterations=1)
    elif method == "Opening":
        return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    else:  # Closing
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

def detect_and_classify(original_bgr, morphed, min_area, classifier):
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = original_bgr.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Classify if a model is available
            if classifier is not None:
                roi = original_bgr[y:y+h, x:x+w]
                if roi.size != 0:
                    feat = extract_features(roi)
                    pred_class = classifier.predict([feat])[0]
                    cv2.putText(output, pred_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return output, count

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.set_page_config(page_title="CV Demo with Training & Classification", layout="centered")
st.title("Interactive Computer Vision Demo")

# Choose mode: Train Classifier or Interactive Demo
mode = st.sidebar.radio("Choose Mode", ["Train Classifier", "Interactive Demo"])

# ---------------------------
# Mode 1: Train Classifier
# ---------------------------
if mode == "Train Classifier":
    st.header("Train a New Classifier")
    st.markdown("""
    Upload a CSV file with annotations (columns: `filename,width,height,class,xmin,ymin,xmax,ymax`)
    and ensure your images are in the specified folder.
    """)
    
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])
    images_folder = st.text_input("Images Folder", value="images")
    
    if csv_file is not None:
        if st.button("Train Classifier"):
            with st.spinner("Training classifier..."):
                clf, report = train_classifier(csv_file, images_folder)
                if clf is not None:
                    save_classifier(clf, "classifier.pkl")
                    st.success("Classifier trained and saved as 'classifier.pkl'")
                    st.text("Classification Report:")
                    st.text(report)
                else:
                    st.error("Training failed. Please check your data and image folder.")
            st.info("After training, switch to 'Interactive Demo' mode to use the classifier.")

# ---------------------------
# Mode 2: Interactive Demo
# ---------------------------
else:
    st.header("Interactive Demo with Pretrained Classifier")
    
    classifier = load_classifier()
    
    # Upload image for processing
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="demo")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original_bgr is not None:
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            gray_img = preprocess_image(original_bgr)
            st.subheader("Original Image")
            st.image(original_rgb, channels="RGB")
            
            # Sidebar controls for processing
            blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", min_value=1, max_value=31, value=5, step=2)
            st.sidebar.header("Threshold Parameters")
            threshold_method = st.sidebar.selectbox("Threshold Method", 
                                                    ["Global", "Otsu", "Adaptive Gaussian", "Adaptive Mean"])
            global_thresh = st.sidebar.slider("Global Threshold", min_value=0, max_value=255, value=127, step=1)
            block_size = st.sidebar.slider("Block Size (Adaptive)", min_value=3, max_value=51, value=11, step=2)
            C = st.sidebar.slider("C (Adaptive)", min_value=-10, max_value=10, value=2, step=1)
            
            st.sidebar.header("Morphology & Filtering")
            morph_method = st.sidebar.selectbox("Morphological Operation", 
                                                ["None", "Erosion", "Dilation", "Opening", "Closing"])
            morph_ksize = st.sidebar.slider("Morph Kernel Size", min_value=1, max_value=21, value=5, step=1)
            
            st.sidebar.header("Contour Detection")
            min_area = st.sidebar.slider("Min Contour Area", min_value=1, max_value=20000, value=1000, step=100)
            
            # Processing pipeline
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            blurred = apply_blur(gray_img, blur_ksize)
            
            thresh = apply_threshold(blurred, threshold_method, global_thresh, block_size, C)
            morphed = apply_morphology(thresh, morph_method, morph_ksize)
            output, count = detect_and_classify(original_bgr, morphed, min_area, classifier)
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            # Display results
            st.subheader("Grayscale Image")
            st.image(gray_img, clamp=True)
            
            st.subheader("Blurred Image")
            st.image(blurred, clamp=True)
            
            st.subheader(f"Thresholded Image ({threshold_method})")
            st.image(thresh, clamp=True)
            
            st.subheader(f"Morphological Operation: {morph_method}")
            st.image(morphed, clamp=True)
            
            st.subheader(f"Final Output – Found {count} Contours")
            st.image(output_rgb, channels="RGB")
        else:
            st.error("Failed to load the image. Please check your file.")
    else:
        st.info("Upload an image to get started in Interactive Demo mode.")
