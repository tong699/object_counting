import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Set the page title and layout
st.set_page_config(page_title="Interactive CV Demo", layout="centered")

st.title("Interactive Computer Vision Demo")

# --------------------------------------------
# 1) Image Upload & Preprocessing
# --------------------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if original_bgr is not None:
        # Convert to RGB for display
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        
        # Show the uploaded image
        st.subheader("Original Image")
        st.image(original_rgb, channels="RGB")
        
        # --------------------------------------------
        # 2) Sidebar Controls
        # --------------------------------------------
        st.sidebar.header("Threshold Parameters")
        threshold_method = st.sidebar.selectbox("Threshold Method", 
                                                ["Global", "Otsu", "Adaptive Gaussian", "Adaptive Mean"])
        
        # Some of these widgets will only matter for certain threshold methods
        global_thresh = st.sidebar.slider("Global Threshold", min_value=0, max_value=255, value=127, step=1)
        
        block_size = st.sidebar.slider("Block Size (Adaptive)", min_value=3, max_value=51, value=11, step=2)
        C = st.sidebar.slider("C (Adaptive)", min_value=-10, max_value=10, value=2, step=1)
        
        st.sidebar.header("Morphology & Filtering")
        morph_method = st.sidebar.selectbox("Morphological Operation", 
                                            ["None", "Erosion", "Dilation", "Opening", "Closing"])
        
        morph_ksize = st.sidebar.slider("Morph Kernel Size", min_value=1, max_value=21, value=5, step=1)
        blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", min_value=1, max_value=31, value=5, step=2)
        
        st.sidebar.header("Contour Detection")
        min_area = st.sidebar.slider("Min Contour Area", min_value=1, max_value=20000, value=1000, step=100)
        
        # --------------------------------------------
        # 3) Processing Pipeline
        # --------------------------------------------
        # (a) Gaussian blur (ensure blur_ksize is odd)
        if blur_ksize < 1:
            blur_ksize = 1
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        
        blurred = cv2.GaussianBlur(gray_img, (blur_ksize, blur_ksize), 0)
        
        # (b) Thresholding
        if threshold_method == "Global":
            _, thresh = cv2.threshold(blurred, global_thresh, 255, cv2.THRESH_BINARY_INV)
        elif threshold_method == "Otsu":
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif threshold_method == "Adaptive Gaussian":
            # Ensure block_size is odd
            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1
            thresh = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           block_size, C)
        else:  # "Adaptive Mean"
            if block_size < 3:
                block_size = 3
            if block_size % 2 == 0:
                block_size += 1
            thresh = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           block_size, C)
        
        # (c) Morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
        if morph_method == "None":
            morphed = thresh
        elif morph_method == "Erosion":
            morphed = cv2.erode(thresh, kernel, iterations=1)
        elif morph_method == "Dilation":
            morphed = cv2.dilate(thresh, kernel, iterations=1)
        elif morph_method == "Opening":
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        else:  # "Closing"
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # (d) Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # (e) Draw bounding boxes on a copy of the original image
        output = original_bgr.copy()  # BGR copy
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Convert output to RGB for display
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        # --------------------------------------------
        # 4) Show results
        # --------------------------------------------
        st.subheader(f"Thresholded Image ({threshold_method})")
        st.image(thresh, clamp=True)  # 'clamp=True' helps ensure it's shown as binary
        
        st.subheader(f"Morphological Operation: {morph_method}")
        st.image(morphed, clamp=True)
        
        st.subheader(f"Final Output – Found {count} Contours")
        st.image(output_rgb, channels="RGB")
        
    else:
        st.write("Failed to load the image. Please check your file.")
else:
    st.write("Upload an image to get started.")
