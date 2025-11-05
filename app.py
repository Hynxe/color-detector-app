import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pyttsx3
import time
import os
from sklearn.cluster import KMeans

# ---------- CONFIG ----------
CSV_PATH = "colors.csv"
STABLE_COLOR_TIME = 2.0
RING_RADIUS = 80
LUMINANCE_THRESHOLD = 130
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
# ----------------------------

st.set_page_config(page_title="Smart Glasses Color Detector", layout="wide")
st.title("🕶️ Smart Glasses Color Detector")

# ---------- Load color data ----------
if not os.path.exists(CSV_PATH):
    st.error("❌ Missing colors.csv. Upload or place it in the same folder.")
    st.stop()
colors = pd.read_csv(CSV_PATH)

# ---------- User options ----------
tts_enabled = st.toggle("🔊 Enable Voice Output (TTS)", value=False)
mode = st.radio("Choose mode:", ["📷 Camera Mode", "🖼️ Image Upload Mode"], horizontal=True)
st.markdown("---")

# ---------- Safe TTS setup ----------
engine = None
if tts_enabled:
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
    except Exception as e:
        st.warning(f"⚠️ TTS unavailable: {e}")
# ------------------------------------

def get_color_name(R, G, B):
    """Return nearest named color from CSV."""
    min_dist = float("inf")
    cname = ""
    for i in range(len(colors)):
        d = abs(R - int(colors.loc[i, "R"])) + abs(G - int(colors.loc[i, "G"])) + abs(B - int(colors.loc[i, "B"]))
        if d < min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]
    return cname

def luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---------- IMAGE UPLOAD MODE ----------
if mode == "🖼️ Image Upload Mode":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Dominant Color"):
            pixels = np.float32(img.reshape((-1, 3)))
            kmeans = KMeans(n_clusters=3, n_init=10)
            kmeans.fit(pixels)
            centers = np.uint8(kmeans.cluster_centers_)
            dominant = centers[0]
            r, g, b = map(int, dominant)
            cname = get_color_name(r, g, b)
            st.success(f"🎨 Dominant Color: **{cname}** ({r},{g},{b})")
            if engine:
                engine.say(cname)
                engine.runAndWait()

# ---------- CAMERA MODE ----------
elif mode == "📷 Camera Mode":
    st.write("🎥 Center your object in view and hold still for ~2 seconds.")
    run_camera = st.button("▶️ Start Camera")
    stop_camera = st.button("⏹️ Stop Camera")

    if run_camera and not stop_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access camera.")
        else:
            placeholder = st.empty()
            last_color = None
            stable_since = None
            spoken_for = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠️ Camera disconnected.")
                    break

                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
                cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

                region = frame[cy-5:cy+5, cx-5:cx+5]
                avg_bgr = region.mean(axis=(0, 1))
                b, g, r = [int(x) for x in avg_bgr]
                cname = get_color_name(r, g, b)

                # Luminance for HUD contrast
                lum = luminance(r, g, b)
                hud_color = (255, 255, 255) if lum < LUMINANCE_THRESHOLD else (0, 0, 0)

                # HUD overlay
                cv2.circle(frame, (cx, cy), RING_RADIUS, hud_color, 3)
                cv2.circle(frame, (cx, cy), 5, hud_color, -1)
                cv2.putText(frame, cname, (cx - 100, cy + RING_RADIUS + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, hud_color, 2, cv2.LINE_AA)

                now = time.time()
                if cname == last_color:
                    if stable_since is None:
                        stable_since = now
                    elapsed = now - stable_since
                    if elapsed >= STABLE_COLOR_TIME:
                        if spoken_for != cname:
                            if engine:
                                engine.say(cname)
                                engine.runAndWait()
                            else:
                                st.write(f"🎨 Detected color: **{cname}**")
                            spoken_for = cname
                else:
                    last_color = cname
                    stable_since = now

                # Show frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Stop loop
                if stop_camera or not run_camera:
                    break

            cap.release()
            placeholder.empty()
            st.success("👋 Camera closed.")