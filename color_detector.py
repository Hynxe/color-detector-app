import cv2
import pandas as pd

# --- Load the image ---
image_path = "test.png"  # change to your file
img = cv2.imread(image_path)

if img is None:
    print("❌ Error: Image not found. Check your path.")
    exit()

# --- Load color dataset ---
csv_path = "colors.csv"
colors = pd.read_csv(csv_path)

# --- Helper to find nearest color ---
def get_color_name(R, G, B):
    min_dist = float('inf')
    cname = ""
    for i in range(len(colors)):
        d = abs(R - colors.loc[i, "R"]) + abs(G - colors.loc[i, "G"]) + abs(B - colors.loc[i, "B"])
        if d < min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]
    return cname

# --- Globals ---
clicked = False
r = g = b = xpos = ypos = 0
color_name = ""

# --- Mouse callback ---
def draw_function(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked, color_name
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        xpos, ypos = x, y
        b, g, r = img[y, x]
        color_name = get_color_name(r, g, b)

# --- Window setup ---
cv2.namedWindow("Color Detector - Press 'q' to quit")
cv2.setMouseCallback("Color Detector - Press 'q' to quit", draw_function)

while True:
    display_img = img.copy()

    if clicked:
        # Draw rectangle with detected color
        cv2.rectangle(display_img, (20, 20), (500, 60), (int(b), int(g), int(r)), -1)

        # Show only color name
        text = f"{color_name}"
        text_color = (255, 255, 255) if (r + g + b) < 400 else (0, 0, 0)
        cv2.putText(display_img, text, (30, 50), 0, 0.8, text_color, 2, cv2.LINE_AA)

    cv2.imshow("Color Detector - Press 'q' to quit", display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting... Goodbye!")
        break

cv2.destroyAllWindows()