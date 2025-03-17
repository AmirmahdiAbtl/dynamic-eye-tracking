# Real-Time Eye Gaze Tracking with Dynamic Calibration

## 📌 Project Overview

This project focuses on **real-time gaze tracking and estimation** using **MediaPipe’s Face Mesh model**. The system dynamically recalibrates gaze positions to account for **head movement, facial scaling, and distance variations**, making gaze tracking more robust and reliable. 

### 🔹 Key Innovation: Artificial Eye Boundary

Instead of storing absolute gaze positions, this method uses an **artificial eye boundary** that dynamically moves and scales with the user’s head. This allows:
1. **Head movement compensation** – The artificial boundary follows head movements, ensuring stable gaze tracking.
2. **Adaptive calibration** – Changes in angle and distance are automatically accounted for by adjusting the boundary size.


## ⚙️ Requirements
Ensure you have the following dependencies installed:
```bash
pip install opencv-python mediapipe numpy matplotlib
```

## 🚀 How to Run
1. **Place your calibration images and test video** inside the project folder.
2. **Run the script:**
   ```bash
   python gaze_tracking.py
   ```
3. The program will:
   - **Calibrate using predefined images**.
   - **Track gaze movements** using a recorded video.
   - **Generate heatmaps, trajectory plots, and quadrant analysis.**

## 🛠️ Methodology

### 1️⃣ Facial Landmark Detection
- Uses **MediaPipe Face Mesh** to detect **468 facial landmarks**.
- Extracts **left eye gaze landmark (468)** for tracking.
- Detects eye boundary and tracks gaze within this dynamic region.

### 2️⃣ Artificial Eye Boundary & Calibration
- Instead of saving **gaze coordinates**, the system saves **relative offsets from an artificial boundary** around the left eye.
- This boundary moves and scales with the user's head, preventing calibration loss.

### 3️⃣ Gaze Tracking & Mapping
- **Detects real-time gaze shifts** by comparing the eye position relative to the artificial boundary.
- **Maps the tracked gaze position** onto a scaled screen coordinate system.

### 4️⃣ Results & Analysis
- **Heatmap Generation**: Shows gaze density across the screen.
- **Quadrant Analysis**: Displays gaze distribution across screen sections.
- **Trajectory Mapping**: Visualizes gaze movement patterns over time.

## ❗ Challenges & Limitations
- **Lighting variations** can impact facial landmark detection.
- **Low-resolution cameras** may introduce tracking noise.
- **Rapid head movements** can cause brief tracking loss.

## 🔮 Future Enhancements
- Implement **CNN-based gaze prediction** for automated calibration.
- Integrate **Kalman filtering** for smoother tracking.
- Use **3D head pose estimation** to improve gaze tracking under rotations.

## 📧 Contact
Author: **Amirmahdi Aboutalebi**  
Email: [Amir.abootalebi2001@gmail.com](mailto:Amir.abootalebi2001@gmail.com)  

### ⭐ If you find this project useful, consider giving it a star on GitHub!
