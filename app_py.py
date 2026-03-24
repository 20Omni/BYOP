import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Medical Image Analyzer", layout="wide")

st.title("🏥 Medical Image Enhancement & Analysis System")

st.write("Upload a medical image (X-ray, scan, etc.) to enhance and analyze it.")

# Upload image
uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert BGR to RGB for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.sidebar.title("🛠️ Processing Options")

    option = st.sidebar.selectbox(
        "Select Operation",
        (
            "Original",
            "Grayscale",
            "Noise Reduction",
            "Contrast Enhancement",
            "Edge Detection",
            "Threshold Segmentation"
        )
    )

    processed = image.copy()

    # Grayscale
    if option == "Grayscale":
        processed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Noise Reduction
    elif option == "Noise Reduction":
        processed = cv2.GaussianBlur(image, (5, 5), 0)

    # Contrast Enhancement
    elif option == "Contrast Enhancement":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        processed = cv2.equalizeHist(gray)

    # Edge Detection
    elif option == "Edge Detection":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        processed = cv2.Canny(gray, 100, 200)

    # Threshold Segmentation
    elif option == "Threshold Segmentation":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, processed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Display side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(processed, caption=option, use_container_width=True)

    # Download option
    st.markdown("### 📥 Download Processed Image")

    if len(processed.shape) == 2:
        processed_to_save = processed
    else:
        processed_to_save = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode(".png", processed_to_save)
    st.download_button(
        label="Download Image",
        data=buffer.tobytes(),
        file_name="processed_image.png",
        mime="image/png"
    )
