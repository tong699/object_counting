import streamlit as st
import cv2
import numpy as np
import time

# --- Page Config ---
st.set_page_config(page_title="Live CV Stream", layout="centered")
st.title("📹 Live Webcam Computer Vision Demo")

# --- Sidebar Controls ---
st.sidebar.header("🛠️ Processing Settings")

blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", 1, 31, 5, step=2)

threshold_method = st.sidebar.selectbox("Threshold Method", 
                                        ["Global", "Otsu", "Adaptive Gaussian", "Adaptive Mean"])
global_thresh = st.sidebar.slider("Global Threshold", 0, 255, 127)
block_size = st.sidebar.slider("Block Size (Adaptive)", 3, 51, 11, step=2)
C = st.sidebar.slider("C (Adaptive)", -10, 10, 2)

morph_method = st.sidebar.selectbox("Morphological Operation", 
                                    ["None", "Erosion", "Dilation", "Opening", "Closing"])
morph_ksize = st.sidebar.slider("Morph Kernel Size", 1, 21, 5)

min_area = st.sidebar.slider("Min Contour Area", 100, 20000, 1000, step=100)

# --- Stream Controls ---
run = st.checkbox("▶️ Start Webcam Stream")
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("🚨 Could not access the webcam.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            # --- Step 1: Grayscale ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Step 2: Gaussian Blur ---
            blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
            blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

            # --- Step 3: Thresholding ---
            if threshold_method == "Global":
                _, thresh = cv2.threshold(blurred, global_thresh, 255, cv2.THRESH_BINARY_INV)
            elif threshold_method == "Otsu":
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif "Adaptive" in threshold_method:
                method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if "Gaussian" in threshold_method else cv2.ADAPTIVE_THRESH_MEAN_C
                block_size = block_size if block_size % 2 == 1 else block_size + 1
                thresh = cv2.adaptiveThreshold(blurred, 255, method, cv2.THRESH_BINARY_INV, block_size, C)

            # --- Step 4: Morphology ---
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
            morphed = thresh.copy()
            if morph_method == "Erosion":
                morphed = cv2.erode(thresh, kernel, iterations=1)
            elif morph_method == "Dilation":
                morphed = cv2.dilate(thresh, kernel, iterations=1)
            elif morph_method == "Opening":
                morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            elif morph_method == "Closing":
                morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # --- Step 5: Contour Detection ---
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output = frame.copy()
            count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    count += 1

            # --- Display Frame ---
            display_frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels="RGB", caption=f"{count} contours found")

            # Small delay to simulate ~30 FPS
            time.sleep(0.03)

        cap.release()
        st.info("Webcam stream ended.")

else:
    st.info("Check the box above to start the webcam.")
