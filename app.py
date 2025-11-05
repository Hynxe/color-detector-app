import streamlit as st
import cv2
import pandas as pd
import numpy as np
import pyttsx3
import time
import os
from sklearn.cluster import KMeans

# ---------- CONFIG ----------
CSV_PATH = "colors.csv"
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
STABLE_COLOR_TIME = 2.0
RING_RADIUS = 80
RING_THICKNESS = 4
ALPHA = 0.45
LUMINANCE_THRESHOLD = 130
# ----------------------------

# Load color CSV
@st.cache_data
def load_colors():
    if not os.path.exists(CSV_PATH):
        st.error("❌ Missing colors.csv in project directory.")
        st.stop()
    return pd.read_csv(CSV_PATH)

colors = load_colors()

# Safe TTS init
def init_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception:
        st.warning("🔇 Voice not available in this environment.")
        return None

engine = init_tts()

def get_color_name(R, G, B):
    min_dist = float('inf')
    cname = ""
    for i in range(len(colors)):
        d = abs(R - int(colors.loc[i, "R"])) + abs(G - int(colors.loc[i, "G"])) + abs(B - int(colors.loc[i, "B"]))
        if d < min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]
    return cname

def luminance(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0722*b

def draw_hud(frame, center, ring_radius, hud_color):
    overlay = frame.copy()
    cx, cy = center
    cv2.circle(overlay, (cx, cy), ring_radius, hud_color, RING_THICKNESS, lineType=cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), max(3, RING_THICKNESS), hud_color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

def detect_dominant_color(frame):
    pixels = np.float32(frame.reshape((-1, 3)))
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)
    centers = np.uint8(kmeans.cluster_centers_)
    r, g, b = map(int, centers[0])
    return r, g, b

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Smart Color Detector", layout="wide")
st.title("🕶️ Smart Glasses Color Detector")
st.caption("Hover, Capture, and Hear the Color — works both locally and online!")

mode = st.radio("Choose mode:", ["📷 Live Camera", "🖼️ Upload Image"])

if mode == "📷 Live Camera":
    st.markdown("### 🎥 Camera Mode")

    # Detect if running in Streamlit Cloud (no hardware camera access)
    in_cloud = "streamlit.app" in st.runtime.scriptrunner.script_run_context.get_script_run_ctx().session_id

    if in_cloud:
        st.info("🌐 Running in Streamlit Cloud — using browser-based camera instead.")
        img = st.camera_input("Take a snapshot to detect color")
        if img:
            file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r, g, b = detect_dominant_color(frame)
            cname = get_color_name(r, g, b)
            st.success(f"🎨 Dominant Color: **{cname}**")
            st.image(frame, caption=f"Detected: {cname}", use_container_width=True)
            if engine:
                engine.say(cname)
                engine.runAndWait()
    else:
        st.info("🎛️ Running locally — using live OpenCV camera. Press 'q' to quit window.")
        start = st.button("▶️ Start Live Camera")
        if start:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access camera. Please check permissions.")
            else:
                last_color = None
                stable_since = None
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
                    cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

                    sample_radius = max(1, RING_RADIUS // 4)
                    region = frame[cy-sample_radius:cy+sample_radius, cx-sample_radius:cx+sample_radius]
                    avg_bgr = region.mean(axis=(0,1))
                    b, g, r = [int(x) for x in avg_bgr]
                    cname = get_color_name(r, g, b)
                    hud_color = (255,255,255) if luminance(r,g,b)<LUMINANCE_THRESHOLD else (0,0,0)
                    draw_hud(frame, (cx,cy), RING_RADIUS, hud_color)
                    cv2.putText(frame, cname, (40,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, hud_color, 3)
                    cv2.imshow("Smart Glasses", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()

elif mode == "🖼️ Upload Image":
    st.markdown("### 🖼️ Upload Image Mode")
    img_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r, g, b = detect_dominant_color(frame)
        cname = get_color_name(r, g, b)
        st.image(frame, caption=f"Detected Dominant Color: {cname}", use_container_width=True)
        st.success(f"🎨 Dominant Color: **{cname}**")
        if engine:
            engine.say(cname)
            engine.runAndWait()