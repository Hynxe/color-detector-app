import cv2
import pandas as pd
import pyttsx3
import time
import os
import sys
import numpy as np

# ---------- CONFIG ----------
CSV_PATH = "colors.csv"
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
STABLE_COLOR_TIME = 2.0   # seconds required to hold color before speaking
RING_RADIUS = 80          # pixels for focus ring size (scales with resolution)
RING_THICKNESS = 4
ALPHA = 0.45              # overlay opacity for the HUD
LUMINANCE_THRESHOLD = 130 # threshold for choosing HUD color (0-255)
# ----------------------------

if not os.path.exists(CSV_PATH):
    print("❌ Missing colors.csv. Put it in the same folder as this script.")
    sys.exit(1)

colors = pd.read_csv(CSV_PATH)

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

def get_color_name(R, G, B):
    """Return nearest named color from CSV based on Manhattan distance in RGB."""
    min_dist = float('inf')
    cname = ""
    for i in range(len(colors)):
        d = abs(R - int(colors.loc[i, "R"])) + abs(G - int(colors.loc[i, "G"])) + abs(B - int(colors.loc[i, "B"]))
        if d < min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]
    return cname

def luminance(r, g, b):
    """Approximate perceived luminance (0-255). Input r,g,b in [0..255]."""
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def draw_glasses_hud(frame, center, ring_radius, hud_color):
    """
    Draws a glasses-style HUD (semi-transparent ring, short cross lines, inner dot).
    We draw on an overlay and alpha-blend.
    """
    overlay = frame.copy()
    cx, cy = center

    # ring
    cv2.circle(overlay, (cx, cy), ring_radius, hud_color, RING_THICKNESS, lineType=cv2.LINE_AA)

    # inner dot
    cv2.circle(overlay, (cx, cy), max(3, RING_THICKNESS), hud_color, -1, lineType=cv2.LINE_AA)

    # short horizontal/vertical ticks (outside ring, subtle)
    tick_len = ring_radius // 3
    thickness = max(1, RING_THICKNESS // 2)
    # top
    cv2.line(overlay, (cx, cy - ring_radius - 6), (cx, cy - ring_radius - 6 - tick_len), hud_color, thickness, lineType=cv2.LINE_AA)
    # bottom
    cv2.line(overlay, (cx, cy + ring_radius + 6), (cx, cy + ring_radius + 6 + tick_len), hud_color, thickness, lineType=cv2.LINE_AA)
    # left
    cv2.line(overlay, (cx - ring_radius - 6, cy), (cx - ring_radius - 6 - tick_len, cy), hud_color, thickness, lineType=cv2.LINE_AA)
    # right
    cv2.line(overlay, (cx + ring_radius + 6, cy), (cx + ring_radius + 6 + tick_len, cy), hud_color, thickness, lineType=cv2.LINE_AA)

    # blend
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

def run_camera_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access camera.")
        return

    # make window fullscreen
    win_name = "Smart Glasses - Center Color (press q to quit)"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_color_name = None
    stable_since = None
    spoken_for = None  # remember last spoken color to avoid repeating immediately

    print("🎥 Smart Glasses active — center your gaze. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Camera frame failed.")
                break

            # mirror, resize to fixed screen size
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

            cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

            # sample a small neighborhood around center to be robust to noise
            sample_radius = max(1, RING_RADIUS // 4)
            y0, y1 = max(0, cy - sample_radius), min(SCREEN_HEIGHT, cy + sample_radius)
            x0, x1 = max(0, cx - sample_radius), min(SCREEN_WIDTH, cx + sample_radius)
            region = frame[y0:y1, x0:x1]

            # compute average color in region (BGR)
            avg_bgr = region.mean(axis=(0,1))
            b, g, r = [int(x) for x in avg_bgr]
            current_name = get_color_name(r, g, b)

            # decide HUD color (auto-contrast) using luminance of center region
            center_lum = luminance(r, g, b)
            hud_color = (255, 255, 255) if center_lum < LUMINANCE_THRESHOLD else (0, 0, 0)

            # draw HUD (glasses-style)
            draw_glasses_hud(frame, (cx, cy), RING_RADIUS, hud_color)

            # draw color preview rectangle & name (top-left)
            preview_w, preview_h = 520, 80
            rect_tl = (20, 20)
            rect_br = (20 + preview_w, 20 + preview_h)
            # fill preview with detected color (BGR)
            cv2.rectangle(frame, rect_tl, rect_br, (b, g, r), -1)
            # pick text color to contrast preview
            txt_color = (255,255,255) if luminance(r,g,b) < LUMINANCE_THRESHOLD else (0,0,0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, current_name, (40, 60), font, 1.4, txt_color, 3, cv2.LINE_AA)

            # speak only if the same color is held for STABLE_COLOR_TIME seconds
            now = time.time()
            if current_name == last_color_name:
                if stable_since is None:
                    stable_since = now
                elapsed = now - stable_since
                # visual feedback: small progress bar under the preview
                bar_x0 = rect_tl[0] + 8
                bar_x1 = rect_br[0] - 8
                bar_y = rect_br[1] - 12
                total_w = bar_x1 - bar_x0
                progress = min(1.0, elapsed / STABLE_COLOR_TIME)
                cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + int(total_w * progress), bar_y + 6), hud_color, -1)
                cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + 6), hud_color, 1, cv2.LINE_AA)
                if elapsed >= STABLE_COLOR_TIME:
                    if spoken_for != current_name:
                        # speak
                        print(f"🎤 {current_name}")
                        engine.say(current_name)
                        engine.runAndWait()
                        spoken_for = current_name
            else:
                last_color_name = current_name
                stable_since = now
                # reset spoken_for only when color changes (prevents immediate repeat)
                # not resetting spoken_for here keeps it from repeating same color quickly
                # to allow repeating after change+return, we'll keep spoken_for as-is

            cv2.imshow(win_name, frame)

            # clean quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.stop()
        print("👋 Exiting — all windows closed.")
        sys.exit(0)

if __name__ == "__main__":
    run_camera_mode()