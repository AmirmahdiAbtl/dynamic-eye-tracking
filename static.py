import cv2 as cv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math

# Paths to your static images and video
top_left_image_path = "top_left.jpg"
bottom_right_image_path = "bottom_right.jpg"
video_path = "gaze_video2.mp4"

# Initialize MediaPipe FaceMesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
)
calibration_points = {"top_left": None, "bottom_right": None}
boundry = {"top_left": None, "bottom_right": None}
calibration_diff = {"top_left": None, "bottom_right": None}

# Helper function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Calibration process for static images
def calibrate_with_image(image_path, corner_label):
    global calibration_points, calibration_diff, boundry
    frame = cv.imread(image_path)
    h, w, _ = frame.shape
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    output = faceMesh.process(frameRGB)

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark
        left_eye_x = int(w * landmarks[468].x)
        left_eye_y = int(h * landmarks[468].y)

        # top_left = (int(landmarks[33].x * w), int(landmarks[159].y * h))
        # bottom_right = (int(landmarks[155].x * w), int(landmarks[144].y * h))
        top_left = (int(landmarks[31].x * w), int(landmarks[30].y * h))
        bottom_right = (int(landmarks[112].x * w), int(landmarks[31].y* h))
        calibration_points[corner_label] = (left_eye_x, left_eye_y)
        if corner_label == "top_left":
            boundry["top_left"] = top_left
            calibration_diff["top_left"] = (boundry["top_left"][0] - calibration_points["top_left"][0] ,
                                            boundry["top_left"][1] - calibration_points["top_left"][1] )
        elif corner_label == "bottom_right":
            boundry["bottom_right"] = bottom_right
            calibration_diff["bottom_right"] = (bottom_right[0] - calibration_points["bottom_right"][0],
                                                bottom_right[1] - calibration_points["bottom_right"][1])

    return frame

# Perform calibration with the two static images
calibrate_with_image(top_left_image_path, "top_left")
calibrate_with_image(bottom_right_image_path, "bottom_right")

print(f"Calibration diff: {calibration_diff} - Calibration points: {calibration_points}, Boundary: {boundry}")

# Compute boundary size
size_boundry = {
    "x": abs(boundry["bottom_right"][0] - boundry["top_left"][0]),
    "y": abs(boundry["bottom_right"][1] - boundry["top_left"][1]),
}

# Process the video for gaze tracking
capture = cv.VideoCapture(video_path)
screen_w, screen_h = 1920, 1080  # Adjust based on your desired screen resolution
left_eye_movements = []

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    h, w, c = frame.shape
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_x = int(w * landmarks[468].x)
        left_y = int(h * landmarks[468].y)

        # top_left = (int(landmarks[33].x * w), int(landmarks[159].y * h))
        # bottom_right = (int(landmarks[155].x * w), int(landmarks[144].y * h))
        top_left = (int(landmarks[31].x * w), int(landmarks[30].y * h))
        bottom_right = (int(landmarks[112].x * w), int(landmarks[31].y* h))
        size_difference = {
            "x": abs(bottom_right[0] - top_left[0]) / size_boundry["x"],
            "y": abs(bottom_right[1] - top_left[1]) / size_boundry["y"],
        }

        calibration_box = {
            "top_left": (top_left[0] - int(calibration_diff["top_left"][0] * size_difference['x']),
                         top_left[1] - int(calibration_diff["top_left"][1] * size_difference['y'])),
            "bottom_right": (bottom_right[0] - int(calibration_diff["bottom_right"][0] * size_difference['x']),
                             bottom_right[1] - int(calibration_diff["bottom_right"][1] * size_difference['y']) - 5),
        }

        h_diff = screen_h / (calibration_box["bottom_right"][1] - calibration_box["top_left"][1])
        w_diff = screen_w / (calibration_box["bottom_right"][0] - calibration_box["top_left"][0])
        current_point = (int((left_x - calibration_box["top_left"][0])),
                         int((left_y - calibration_box["top_left"][1])))
        left_eye_movements.append((current_point[0] * int(w_diff), current_point[1] * int(h_diff)))

        cv.rectangle(frame, calibration_box["top_left"], calibration_box["bottom_right"], (255, 255, 255), 2)
        cv.putText(frame, ".", (left_x, left_y), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv.imshow("Eye Tracking", frame)

    if cv.waitKey(20) & 0xFF == ord("q"):  # Press 'q' to quit
        break

capture.release()
cv.destroyAllWindows()

# Generate the heatmap
heatmap, x_edges, y_edges = np.histogram2d(
    [point[0] for point in left_eye_movements],
    [point[1] for point in left_eye_movements],
    bins=[100, 100],
    range=[[0, screen_w], [0, screen_h]],
)

plt.figure(figsize=(10, 6))
plt.imshow(heatmap.T, origin='lower', extent=[0, screen_w, 0, screen_h], cmap='hot', interpolation='nearest')
plt.title("Heatmap of Gaze Points")
plt.xlabel("X Coordinate (Screen Pixels)")
plt.ylabel("Y Coordinate (Screen Pixels)")
plt.colorbar(label="Gaze Density")
plt.gca().invert_yaxis()
plt.show()

# Perform quadrant analysis
center_x = screen_w // 2
center_y = screen_h // 2
quadrant_counts = {"Top-Left": 0, "Top-Right": 0, "Bottom-Left": 0, "Bottom-Right": 0}

for point in left_eye_movements:
    if point[0] < center_x and point[1] < center_y:
        quadrant_counts["Top-Left"] += 1
    elif point[0] >= center_x and point[1] < center_y:
        quadrant_counts["Top-Right"] += 1
    elif point[0] < center_x and point[1] >= center_y:
        quadrant_counts["Bottom-Left"] += 1
    elif point[0] >= center_x and point[1] >= center_y:
        quadrant_counts["Bottom-Right"] += 1

# Plot the bar chart
plt.figure(figsize=(8, 6))
plt.bar(quadrant_counts.keys(), quadrant_counts.values(), color='orange')
plt.title("Gaze Time in Each Screen Quadrant")
plt.xlabel("Screen Quadrant")
plt.ylabel("Count of Gaze Points")
plt.show()

# Calculate distances between consecutive points
distances = [calculate_distance(left_eye_movements[i], left_eye_movements[i-1])
             for i in range(1, len(left_eye_movements))]
time_steps = range(1, len(left_eye_movements))  

plt.figure(figsize=(10, 6))
plt.plot(time_steps, distances, marker='o', linestyle='-', color='g')
plt.title("Distance Traveled by Gaze Over Time")
plt.xlabel("Frame Number (Time)")
plt.ylabel("Distance (Pixels)")
plt.grid(True)
plt.show()


# Display gaze trajectory
white_image = np.ones((screen_h, screen_w, 3), np.uint8) * 255

for i in range(1, len(left_eye_movements)):
    start_point = left_eye_movements[i - 1]
    end_point = left_eye_movements[i]
    
    # Draw the arrow
    cv.arrowedLine(white_image, start_point, end_point, (0, 0, 255), 1, tipLength=0.1)
    
    # Compute the midpoint of the arrow
    midpoint = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
    
    # Display the step number at the midpoint of the arrow
    step_number = str(i)  # Step number starts from 1
    font_scale = 0.4  # Smaller text size
    thickness = 1
    if i % 50 == 0:
        cv.putText(white_image, step_number, midpoint, cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

cv.imshow("Eye Movements", white_image)
output_path = "modified_image.jpg"  # You can specify your own file path
cv.imwrite(output_path, white_image)
cv.waitKey(0)
cv.destroyAllWindows()
