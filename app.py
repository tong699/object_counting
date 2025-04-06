import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import av

st.set_page_config(page_title="Live CV Demo", layout="centered")
st.title("🎥 Live Computer Vision with Real-Time Processing")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Processing Options")

blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", 1, 31, 5, step=2)
threshold_method = st.sidebar.selectbox("Threshold Method", ["Global", "Otsu", "Adaptive Gaussian", "Adaptive Mean"])
global_thresh = st.sidebar.slider("Global Threshold", 0, 255, 127)
block_size = st.sidebar.slider("Adaptive Block Size", 3, 51, 11, step=2)
C = st.sidebar.slider("Adaptive C", -10, 10, 2)
morph_method = st.sidebar.selectbox("Morph Operation", ["None", "Erosion", "Dilation", "Opening", "Closing"])
morph_ksize = st.sidebar.slider("Morph Kernel Size", 1, 21, 5)
min_area = st.sidebar.slider("Min Contour Area", 100, 20000, 1000, step=100)

# -----------------------------
# Frame Processor
# -----------------------------
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # Thresholding
        if threshold_method == "Global":
            _, thresh = cv2.threshold(blurred, global_thresh, 255, cv2.THRESH_BINARY_INV)
        elif threshold_method == "Otsu":
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif threshold_method == "Adaptive Gaussian":
            block = block_size if block_size % 2 == 1 else block_size + 1
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, block, C)
        else:  # Adaptive Mean
            block = block_size if block_size % 2 == 1 else block_size + 1
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, block, C)

        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
        if morph_method == "Erosion":
            morphed = cv2.erode(thresh, kernel, iterations=1)
        elif morph_method == "Dilation":
            morphed = cv2.dilate(thresh, kernel, iterations=1)
        elif morph_method == "Opening":
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        elif morph_method == "Closing":
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        else:
            morphed = thresh

        # Contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img

# -----------------------------
# Start WebRTC Stream
# -----------------------------
webrtc_streamer(
    key="live-cv",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
