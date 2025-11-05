⸻

🕶️ Smart Glasses Color Assistant

This project helps colorblind users detect and hear the true colors of objects through a camera feed — like a prototype for smart color-detecting glasses.

⸻

🧠 Features
	•	🎥 Live camera color detection
Detects the color in the center crosshair in real-time.
	•	🔊 Speaks the detected color name aloud
Only announces a color after it’s stable for ~2 seconds (reduces random noise).
	•	⏹️ Clean exit
Press Q anytime to close both the camera window and terminal safely.
	•	🎯 Improved crosshair overlay
Designed to look like a targeting assist (simulating smart glasses).

⸻

🗂️ Folder Structure

COLOR/
├── smart_glasses_camera.py   # Main program (run this)
├── color_detector.py         # Helper for color matching
├── colors.csv                # Color dataset (CSS3 colors)
└── README.md                 # (this file)


⸻

⚙️ Requirements
You need Python 3.9+ and these packages:

pip install opencv-python pyttsx3 pandas numpy scikit-learn


⸻

▶️ Run the Program
In the terminal (inside the COLOR folder):

python smart_glasses_camera.py

Then:
	•	Point the camera at an object
	•	Hold still for about 2 seconds
	•	The app will say the color name aloud
	•	Press Q to quit cleanly

⸻

💡 Future Ideas
	•	Add Bluetooth audio output for glasses speakers
	•	Connect to Raspberry Pi + camera module
	•	Integrate ambient light correction
	•	Add dominant color detection mode

⸻

Would you like me to format it so it automatically displays colored emoji text in the terminal too (for a nicer visual effect when running)?