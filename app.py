import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Smart Color Detector", layout="centered")

st.title("🎨 Smart Color Detector")
st.write("Choose a mode below to start detecting colors!")

# --- Mode buttons ---
mode = st.radio(
    "Select mode:",
    ["Camera Mode", "Image Mode"],
    horizontal=True
)

st.markdown("---")

# --- Camera Mode ---
if mode == "Camera Mode":
    st.subheader("📷 Real-Time Camera Color Detection")
    st.write("Press **Start Camera** to begin and **Q** to quit.")

    if st.button("Start Camera"):
        subprocess.run(["python3", "smart_glasses_camera.py"])

# --- Image Mode ---
elif mode == "Image Mode":
    st.subheader("🖼️ Detect Colors from Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        save_path = "uploaded_image.png"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(save_path, caption="Uploaded Image", use_container_width=True)
        st.write("Press **Start Image Detector** to click on colors.")
        if st.button("Start Image Detector"):
            subprocess.run(["python3", "color_detector.py"])

st.markdown("---")
st.write("👋 Press **Q** in the OpenCV window to close and return here.")