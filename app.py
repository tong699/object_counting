import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Real-Time Malaysian Coin Detection")
st.write("Adjust parameters from the sidebar to tune the coin detection algorithm.")

# Sidebar parameters for Hough Circle Transform tuning
dp = st.sidebar.slider("dp (×10)", min_value=1, max_value=30, value=12) / 10.0
minDist = st.sidebar.slider("minDist", min_value=10, max_value=200, value=50)
param1 = st.sidebar.slider("param1", min_value=10, max_value=200, value=50)
param2 = st.sidebar.slider("param2", min_value=10, max_value=100, value=30)
minRadius = st.sidebar.slider("minRadius", min_value=1, max_value=100, value=10)
maxRadius = st.sidebar.slider("maxRadius", min_value=1, max_value=200, value=50)

class CoinDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize with default parameters; these will be updated via the sliders
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.minRadius = minRadius
        self.maxRadius = maxRadius

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Update parameters dynamically from Streamlit sidebar
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.minRadius = minRadius
        self.maxRadius = maxRadius

        # Convert the frame to a numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using the Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.minDist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.minRadius,
            maxRadius=self.maxRadius,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            cv2.putText(img, f"Detected: {len(circles)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        else:
            cv2.putText(img, "No circles detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img

# Deploy the video stream using streamlit-webrtc
webrtc_streamer(
    key="coin-detection",
    video_transformer_factory=CoinDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
