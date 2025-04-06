import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --------------------------
# Page Config & Title
# --------------------------
st.set_page_config(page_title="Interactive CV Toolkit", layout="centered")
st.title("🧠 Interactive Computer Vision Toolkit")

# --------------------------
# Image Upload
# --------------------------
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original_bgr is not None:
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

        st.image(original_rgb, caption="🖼️ Original Image", use_column_width=True)

        # --------------------------
        # Sidebar Controls
        # --------------------------
        st.sidebar.header("🛠️ Preprocessing")
        blur_ksize = st.sidebar.slider("Gaussian Blur Kernel", 1, 31, 5, step=2)

        st.sidebar.header("⚡ Thresholding")
        threshold_method = st.sidebar.selectbox("Method", ["Global", "Otsu", "Adaptive Gaussian", "Adaptive Mean"])
        global_thresh = st.sidebar.slider("Global Threshold", 0, 255, 127)
        block_size = st.sidebar.slider("Block Size (Adaptive)", 3, 51, 11, step=2)
        C = st.sidebar.slider("C (Adaptive)", -10, 10, 2)

        st.sidebar.header("🔁 Morphology")
        morph_method = st.sidebar.selectbox("Operation", ["None", "Erosion", "Dilation", "Opening", "Closing"])
        morph_ksize = st.sidebar.slider("Kernel Size", 1, 21, 5)

        st.sidebar.header("🔍 Contour Detection")
        min_area = st.sidebar.slider("Minimum Contour Area", 100, 20000, 1000, step=100)
        show_labels = st.sidebar.checkbox("Show Contour Labels", value=True)
        label_type = st.sidebar.radio("Label Type", ["ID", "Area"], horizontal=True)

        st.sidebar.header("🧱 Edge Detection")
        apply_canny = st.sidebar.checkbox("Use Canny Edge Detection")
        canny_thresh1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
        canny_thresh2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)

        st.sidebar.header("🧬 Connected Components")
        use_components = st.sidebar.checkbox("Show Connected Components")

        st.sidebar.header("🎯 ROI Selection")
        enable_crop = st.sidebar.checkbox("Enable ROI Drawing")

        # --------------------------
        # Preprocessing
        # --------------------------
        blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.GaussianBlur(gray_img, (blur_ksize, blur_ksize), 0)

        st.image(gray_img, caption="Grayscale", clamp=True)
        st.image(blurred, caption="Blurred", clamp=True)

        # --------------------------
        # Thresholding / Edges
        # --------------------------
        if apply_canny:
            edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
            st.image(edges, caption="Canny Edges", clamp=True)
            thresh = edges.copy()
        else:
            if threshold_method == "Global":
                _, thresh = cv2.threshold(blurred, global_thresh, 255, cv2.THRESH_BINARY_INV)
            elif threshold_method == "Otsu":
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif "Adaptive" in threshold_method:
                block_size = block_size if block_size % 2 == 1 else block_size + 1
                method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if threshold_method.endswith("Gaussian") else cv2.ADAPTIVE_THRESH_MEAN_C
                thresh = cv2.adaptiveThreshold(blurred, 255, method, cv2.THRESH_BINARY_INV, block_size, C)
            st.image(thresh, caption=f"Thresholded ({threshold_method})", clamp=True)

        # --------------------------
        # Morphology
        # --------------------------
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

        st.image(morphed, caption=f"Morphology: {morph_method}", clamp=True)

        # --------------------------
        # Contours & Labels
        # --------------------------
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = original_bgr.copy()
        count = 0

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if show_labels:
                    label = f"{i+1}" if label_type == "ID" else f"{int(area)}"
                    cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                count += 1

        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        st.subheader(f"✅ Final Output – {count} Contours Found")
        st.image(output_rgb, use_column_width=True)

        # --------------------------
        # Connected Components
        # --------------------------
        if use_components:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed, connectivity=8)
            cc_output = original_bgr.copy()
            for i in range(1, num_labels):  # Skip background
                x, y, w, h, area = stats[i]
                if area >= min_area:
                    cv2.rectangle(cc_output, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(cc_output, f"ID:{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            st.subheader(f"🧬 Connected Components: {num_labels - 1}")
            st.image(cv2.cvtColor(cc_output, cv2.COLOR_BGR2RGB), use_column_width=True)

        # --------------------------
        # ROI Selection
        # --------------------------
        if enable_crop:
            st.subheader("🎯 Draw ROI on Image")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.3)",
                stroke_width=2,
                background_image=Image.fromarray(original_rgb),
                update_streamlit=True,
                height=original_rgb.shape[0],
                width=original_rgb.shape[1],
                drawing_mode="rect",
                key="canvas",
            )

            if canvas_result.json_data and canvas_result.json_data["objects"]:
                obj = canvas_result.json_data["objects"][0]
                left = int(obj["left"])
                top = int(obj["top"])
                width = int(obj["width"])
                height = int(obj["height"])

                roi = original_rgb[top:top+height, left:left+width]
                st.image(roi, caption="🖼️ Selected ROI", use_column_width=True)

        # --------------------------
        # Download Final Output
        # --------------------------
        result_pil = Image.fromarray(output_rgb)
        st.download_button("💾 Download Processed Image", data=result_pil.tobytes("jpeg", "RGB"), file_name="processed_result.jpg")

    else:
        st.error("❌ Failed to load the image. Try again.")
else:
    st.info("👆 Upload an image to begin.")
