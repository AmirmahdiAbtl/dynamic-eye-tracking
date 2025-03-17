import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
import math
# Load the image
capture = cv.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
left_eye_movements = []
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh( 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
    )
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

calibration_points = { 
    "top_left": None, "bottom_right": None,
}
calibration_diff = {
    "top_left": None, "bottom_right": None,
}
boundry = {
    "top_left": None, "bottom_right": None,
}
calibration_complete = False
previous_point = None
def calibrate_corners():
    global calibration_points, calibration_complete
    prompts = ["top_left", 'bottom_right']
    for corner in prompts:
        print(f"Please look at the {corner} corner of the screen.")
        while True:
            _, frame = capture.read()
            frame = cv.flip(frame, 1)
            h, w, _ = frame.shape
            frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            output = faceMesh.process(frameRGB)

            if output.multi_face_landmarks:
                landmarks = output.multi_face_landmarks[0].landmark
                
                left_eye_x = int(w * landmarks[468].x)
                left_eye_y = int(h * landmarks[468].y)
                # print(left_eye_x, left_eye_y)
                top_left = (int(landmarks[33].x * w), int(landmarks[159].y * h))
                bottom_right = (int(landmarks[155].x * w), int(landmarks[144].y* h))

                cv.putText(frame, f"Look at the {corner} and press 'c'", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow("Calibration", frame)

                if cv.waitKey(1) & 0xFF == ord('c'):
                    calibration_points[corner] = (left_eye_x, left_eye_y)
                    if corner == "top_left":
                        boundry["top_left"] = top_left
                        boundry["bottom_right"] = bottom_right
                        calibration_diff["top_left"] = (boundry["top_left"][0] - calibration_points["top_left"][0], boundry["top_left"][1] - calibration_points["top_left"][1])
                    if corner == "bottom_right":
                        calibration_diff["bottom_right"] = (bottom_right[0] - calibration_points["bottom_right"][0], bottom_right[1] - calibration_points["bottom_right"][1])
                    break

    calibration_complete = True
    cv.destroyAllWindows()
calibrate_corners()
print(f"Calibration diff : , {calibration_diff} - calibration points: {calibration_points}, boundry: {boundry}")
size_boundry = {"x": abs(boundry["bottom_right"][0] - boundry["top_left"][0]), "y": abs(boundry["bottom_right"][1] - boundry["top_left"][1])}

each_time_boundry = {
    "top_left": None, "bottom_right": None,
}
while True:
    _, frame = capture.read()
    frame = cv.flip(frame, 1)
    h , w , c = frame.shape
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = faceMesh.process(frameRGB)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_x = int(w * landmarks[468].x)
        left_y = int(h * landmarks[468].y)

        top_left = (int(landmarks[33].x * w), int(landmarks[159].y * h))
        bottom_right = (int(landmarks[155].x * w), int(landmarks[144].y* h))

        size_difference = {"x": abs(bottom_right[0] - top_left[0]) / size_boundry["x"], "y": abs(bottom_right[1] - top_left[1]) / size_boundry["y"]}
        
        calibration_box = {"top_left" : (top_left[0] - int(calibration_diff["top_left"][0] * size_difference['x']), 
                                         top_left[1] - int(calibration_diff["top_left"][1] * size_difference['y'])), 
                           "bottom_right": (bottom_right[0] - int(calibration_diff["bottom_right"][0] * size_difference['x']), 
                                            bottom_right[1] - int(calibration_diff["bottom_right"][1] * size_difference['y']))}

        h_diff = screen_h / (calibration_box["bottom_right"][1] - calibration_box["top_left"][1])
        w_diff = screen_w / (calibration_box["bottom_right"][0] - calibration_box["top_left"][0])
        current_point = (int((left_x - calibration_box["top_left"][0])), int((left_y - calibration_box["top_left"][1])))
        left_eye_movements.append((current_point[0] * int(w_diff), current_point[1]* int(h_diff)))
        print(f"Current Point: {(left_x, left_y)} - Calibration Box: {calibration_box} - Current Point: {current_point}")
        # print(f"Calibration box : {calibration_box}")
        # print(current_point)
        # print(f"Each Time Boundry: {each_time_boundry}")
        cv.rectangle(frame, calibration_box["top_left"], calibration_box["bottom_right"], (255, 255, 255), 2)
        # cv.rectangle(frame, calibration_points["top_left"], calibration_points["bottom_right"], (0, 255, 255), 2)
        cv.rectangle(frame, top_left, bottom_right, (255, 2, 255), 2)
        cv.putText(frame, ".", (int(landmarks[468].x * w) , int(landmarks[468].y * h) ), cv.FONT_HERSHEY_PLAIN, 2, (255, 2, 255), 2)
        
        # print(top_left, bottom_right)
        cv.imshow("Eye Tracking", frame)
    if cv.waitKey(20) & 0xFF == ord("d"):
        break

capture.release()
cv.destroyAllWindows()
# Plot the trajectory of gaze points
# Create a heatmap of gaze points
heatmap, x_edges, y_edges = np.histogram2d(
    [point[0] for point in left_eye_movements],
    [point[1] for point in left_eye_movements],
    bins=[100, 100], 
    range=[[0, screen_w], [0, screen_h]]
)

plt.figure(figsize=(10, 6))
plt.imshow(heatmap.T, origin='lower', extent=[0, screen_w, 0, screen_h], cmap='hot', interpolation='nearest')
plt.title("Heatmap of Gaze Points")
plt.xlabel("X Coordinate (Screen Pixels)")
plt.ylabel("Y Coordinate (Screen Pixels)")
plt.colorbar(label="Gaze Density")
plt.gca().invert_yaxis()
plt.show()


cv.waitKey(0)
cv.destroyAllWindows()


print(h_diff, w_diff)
white_image = np.ones((screen_h, screen_w, 3), np.uint8) * 255

for i in range(len(left_eye_movements)):
    cv.arrowedLine(white_image, left_eye_movements[i-1], left_eye_movements[i], (0, 0, 255), 1, tipLength=0.1)
    # print(x, y)
cv.imshow("Eye Movements", white_image)
cv.waitKey(0)
cv.destroyAllWindows()