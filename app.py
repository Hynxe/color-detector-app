import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import pyttsx3

# ---------- CONFIG ----------
CSV_PATH = "colors.csv"
STABLE_HOVER_TIME = 2.0
LUMINANCE_THRESHOLD = 130
# ----------------------------

# Load color dataset
colors = pd.read_csv(CSV_PATH)

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

# ---------- Helper Functions ----------
def get_color_name(R, G, B):
    """Return nearest named color from CSV based on Manhattan distance in RGB."""
    min_dist = float("inf")
    cname = ""
    for i in range(len(colors)):
        d = abs(R - int(colors.loc[i, "R"])) + abs(G - int(colors.loc[i, "G"])) + abs(B - int(colors.loc[i, "B"]))
        if d < min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]
    return cname

def luminance(r, g, b):
    """Approximate perceived luminance."""
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# ---------- Streamlit Setup ----------
st.set_page_config(page_title="Smart Color Detector", layout="wide")
st.title("🎨 Smart Color Detector")

mode = st.radio("Choose Mode:", ["📸 Camera Mode", "🖼️ Image Upload Mode"], horizontal=True)
st.markdown("---")

# =======================================
# MODE 1: CAMERA DETECTOR
# =======================================
if mode == "📸 Camera Mode":
    st.subheader("Live Camera Color Detection")
    st.write("Center a color under the crosshair and hold for 2 seconds to hear its name.")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        last_color = None
        stable_since = time.time()
        spoken = None

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Unable to access camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2

            # Sample a small region around center
            region = frame[cy - 10:cy + 10, cx - 10:cx + 10]
            avg_bgr = region.mean(axis=(0, 1))
            b, g, r = [int(x) for x in avg_bgr]
            current_name = get_color_name(r, g, b)

            # Choose HUD color (white or black depending on background)
            hud_color = (255, 255, 255) if luminance(r, g, b) < LUMINANCE_THRESHOLD else (0, 0, 0)

            # Draw HUD crosshair
            cv2.circle(frame, (cx, cy), 60, hud_color, 3)
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), hud_color, 2)
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), hud_color, 2)

            # Draw color preview box
            preview_h, preview_w = 80, 300
            cv2.rectangle(frame, (20, 20), (20 + preview_w, 20 + preview_h), (b, g, r), -1)
            txt_color = (255, 255, 255) if luminance(r, g, b) < LUMINANCE_THRESHOLD else (0, 0, 0)
            cv2.putText(frame, current_name, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 3)

            # Stability check for voice feedback
            now = time.time()
            if current_name == last_color:
                if now - stable_since > STABLE_HOVER_TIME and spoken != current_name:
                    engine.say(current_name)
                    engine.runAndWait()
                    spoken = current_name
            else:
                last_color = current_name
                stable_since = now

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        st.write("Camera stopped.")

# =======================================
# MODE 2: IMAGE UPLOAD
# =======================================
elif mode == "🖼️ Image Upload Mode":
    st.subheader("Upload an Image to Detect Colors")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, channels="RGB", caption="Uploaded Image")

        st.write("Click anywhere on the image to detect color (experimental).")

        # Streamlit can’t capture pixel clicks directly yet,
        # so we’ll add a simple hover color picker area for now.
        x = st.slider("X (horizontal position)", 0, image.shape[1]-1, image.shape[1]//2)
        y = st.slider("Y (vertical position)", 0, image.shape[0]-1, image.shape[0]//2)

        pixel = image[y, x]
        r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
        cname = get_color_name(r, g, b)
        st.write(f"**Detected Color:** {cname}")
        st.markdown(
            f"<div style='width:100px;height:50px;background-color:rgb({r},{g},{b});border-radius:8px;'></div>",
            unsafe_allow_html=True
        )