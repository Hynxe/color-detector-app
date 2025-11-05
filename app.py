import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd

# =========================
# Safe TTS import (local only)
# =========================
try:
    import pyttsx3
    tts_available = True
except:
    tts_available = False


# =========================
# Load Colors
# =========================
@st.cache_data
def load_colors():
    df = pd.read_csv("colors.csv")
    return df


colors = load_colors()


# =========================
# Utility functions
# =========================
def get_color_name(R, G, B):
    """Find closest color name from dataset"""
    minimum = 10000
    cname = ""
    for i in range(len(colors)):
        d = abs(R - int(colors.loc[i, "R"])) + abs(G - int(colors.loc[i, "G"])) + abs(B - int(colors.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = colors.loc[i, "color_name"]
    return cname


def detect_dominant_color(image):
    """Detect dominant color in an image using KMeans"""
    img = np.array(image)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(img)
    cluster_centers = np.uint8(kmeans.cluster_centers_)
    dominant_color = cluster_centers[np.argmax(np.bincount(kmeans.labels_))]
    return tuple(dominant_color)


def speak_text(text):
    """Speak detected color aloud (local only)"""
    if tts_available and "STREAMLIT_RUNTIME" not in os.environ:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass


# =========================
# Streamlit App Layout
# =========================
st.set_page_config(page_title="Smart Color Detector", layout="wide")
st.title("🎨 Smart Color Detector")

st.sidebar.markdown("## Choose Mode")
mode = st.sidebar.radio("Select an option:", ["🖼️ Upload Mode", "📷 Camera Mode"])

# =========================
# Upload Mode
# =========================
if mode == "🖼️ Upload Mode":
    st.subheader("🖼️ Upload an Image to Detect Colors")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Dominant Color"):
            color = detect_dominant_color(image)
            cname = get_color_name(*color)
            st.markdown(f"### 🎯 Dominant Color: `{cname}` ({color})")

            color_preview = np.zeros((100, 300, 3), np.uint8)
            color_preview[:, :] = color[::-1]  # Convert RGB → BGR
            st.image(color_preview, caption=f"Preview: {cname}", use_container_width=False)

            speak_text(f"The dominant color is {cname}")

# =========================
# Camera Mode
# =========================
elif mode == "📷 Camera Mode":
    st.subheader("🎥 Camera Mode")

    # Detect if app is running in the cloud (no hardware camera)
    in_cloud = "STREAMLIT_RUNTIME" in os.environ

    if in_cloud:
        st.info("🌐 Running in Streamlit Cloud — using browser-based camera input.")
        picture = st.camera_input("Take a photo using your device camera")

        if picture is not None:
            image = Image.open(picture)
            st.image(image, caption="Captured Image", use_container_width=True)

            color = detect_dominant_color(image)
            cname = get_color_name(*color)
            st.markdown(f"### 🎯 Dominant Color: `{cname}` ({color})")

            color_preview = np.zeros((100, 300, 3), np.uint8)
            color_preview[:, :] = color[::-1]
            st.image(color_preview, caption=f"Preview: {cname}", use_container_width=False)

            speak_text(f"The dominant color is {cname}")
    else:
        st.warning("📸 Local Mode — using your computer’s webcam.")
        st.info("Make sure your camera is connected and permissions are granted.")

        start_cam = st.button("Start Camera")
        if start_cam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access camera.")
            else:
                st.success("✅ Camera started! Press 'q' to quit in the local window.")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imshow("Smart Color Detector - Press 'q' to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                cap.release()
                cv2.destroyAllWindows()