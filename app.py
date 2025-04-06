import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Page setup
st.set_page_config(page_title="Interactive CV Demo", layout="centered")
st.title("🧠 Interactive Computer Vision Demo")

# --------------------------------------------
# Image Upload
# --------------------------------------------
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original_bgr is not None:
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

        st.image(original_rgb, caption="Original Image", use_column_width=True)

        # --------------------------------------------
        # Sidebar Controls
        # --------------------------------------------
        with st.sidebar:
            st.header("🛠️ Preprocessing")
            blur_ksize = st.slider("Gaussian Blur Kernel", 1, 31, 5, step=2)

            st.header("⚡ Thresholding")
            threshold_method = st.selectbox("Method", ["Global", "Otsu", "Adaptive Gaussian", "Adaptive Mean"])
            global_thresh = st.slider("Global Threshold", 0, 255, 127)
            block_size = st.slider("Block Size (Adaptive)", 3, 51, 11, step=2)
            C = st.slider("C (Adaptive)", -10, 10, 2)

            st.header("🔁 Morphology")
            morph_method = st.selectbox("Operation", ["None", "Erosion", "Dilation", "Opening", "Closing"])
            morph_ksize = st.slider("Kernel Size", 1, 21, 5)

            st.header("🔍 Contours")
            min_area = st.slider("Minimum Contour Area", 100, 20000, 1000, step=100)

        # --------------------------------------------
        # Image Processing Pipeline
        # --------------------------------------------
        # (a) Gaussian blur
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.GaussianBlur(gray_img, (blur_ksize, blur_ksize), 0)

        # (b) Thresholding
        if threshold_method == "Global":
            _, thresh = cv2.threshold(blurred, global_thresh, 255, cv2.THRESH_BINARY_INV)
        elif threshold_method == "Otsu":
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif "Adaptive" in threshold_method:
            block_size = block_size if block_size % 2 == 1 else block_size + 1
            method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if threshold_method.endswith("Gaussian") else cv2.ADAPTIVE_THRESH_MEAN_C
            thresh = cv2.adaptiveThreshold(blurred, 255, method, cv2.THRESH_BINARY_INV, block_size, C)

        # (c) Morphology
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

        # (d) Contour detection
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = original_bgr.copy()
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                count += 1

        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # --------------------------------------------
        # Display Results
        # --------------------------------------------
        with st.expander("📷 Grayscale & Blurred Images"):
            st.image(gray_img, caption="Grayscale", clamp=True)
            st.image(blurred, caption="Blurred", clamp=True)

        with st.expander("🧾 Thresholding & Morphology"):
            st.image(thresh, caption=f"Thresholded ({threshold_method})", clamp=True)
            st.image(morphed, caption=f"Morphology: {morph_method}", clamp=True)

        st.subheader(f"✅ Final Output – {count} Contours Found")
        if count == 0:
            st.warning("No contours found with the current settings.")
        st.image(output_rgb, channels="RGB", use_column_width=True)

        # --------------------------------------------
        # Download
        # --------------------------------------------
        result_pil = Image.fromarray(output_rgb)
        st.download_button("💾 Download Result", data=result_pil.tobytes("jpeg", "RGB"), file_name="processed_result.jpg")
        
    else:
        st.error("❌ Failed to load the image. Try again.")
else:
    st.info("👆 Upload an image to get started.")
