import streamlit as st
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import pyttsx3

# =========================
# Load color data
# =========================
@st.cache_data
def load_colors():
    colors = pd.read_csv("colors.csv")
    return colors

colors = load_colors()

# =========================
# Utility functions
# =========================
def get_color_name(R, G, B):
    """Find the closest color name from dataset."""
    minimum = float('inf')
    cname = "Unknown"
    for i in range(len(colors)):
        d = abs(R - int(colors.loc[i, "R"])) + abs(G - int(colors.loc[i, "G"])) + abs(B - int(colors.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = colors.loc[i, "color_name"]
    return cname

def get_dominant_color(image, k=3):
    """Use KMeans to find the dominant color in an image."""
    img = np.array(image)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img)
    cluster_centers = np.uint8(kmeans.cluster_centers_)
    counts = np.bincount(kmeans.labels_)
    dominant = cluster_centers[np.argmax(counts)]
    return tuple(int(c) for c in dominant)

def speak_color(color_name):
    """Speak the detected color name (local only)."""
    try:
        engine = pyttsx3.init()
        engine.say(color_name)
        engine.runAndWait()
    except:
        st.warning("Speech not supported on this platform.")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Smart Color Camera", page_icon="🎥", layout="centered")

st.title("🎥 Smart Color Detector (Camera Mode)")
st.write("Use your **camera** to capture an image and detect the dominant color.")

# Camera input
camera_image = st.camera_input("Capture Image")

if camera_image:
    # Convert to RGB image
    image = Image.open(camera_image).convert('RGB')
    st.image(image, caption="Captured Frame", use_container_width=True)

    # Detect dominant color
    dominant = get_dominant_color(image)
    color_name = get_color_name(*dominant)

    # Display result
    st.subheader("🎨 Detected Dominant Color")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Name:** {color_name}")
        st.markdown(f"**RGB:** {dominant}")
    with col2:
        st.markdown(
            f"""
            <div style='background-color: rgb{dominant}; 
                        width: 100%; height: 100px; 
                        border-radius: 10px;'>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("🔊 Speak Color"):
        speak_color(color_name)
else:
    st.info("📸 Click the camera above to capture an image.")