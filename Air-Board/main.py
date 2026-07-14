import numpy as np
import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
# # fetched from curl -o hand_landmarker.task \
# #  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
MODEL_PATH = "/Users/bhavan/Documents/codes/COMPUTER_VISION/Air-Board/hand_landmarker.task"

cap = cv2.VideoCapture(0)

ret,frame = cap.read()
if not ret:
    print("Camera not aceessible")
    exit()

height,width, _ = frame.shape

canvas = np.zeros((height, width, 3), dtype=np.uint8)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), # connecting thumb landmarks
    (0,5),(5,6),(6,7),(7,8), # connecting index finger landmarks
    (5,9),(9,10),(10,11),(11,12), # connecting middle finger landmarks
    (9,13),(13,14),(14,15),(15,16), # connecting ring finger landmarks
    (13,17),(17,18),(18,19),(19,20), # connecting pinky finger landmarks
    (0,17) # connecting palm landmarks
]



def draw_landmark_and_connect(frame, hand_landmarker, width, height):
    points = {}
    for index, value in enumerate(hand_landmarker):
        x = int(value.x*width)
        y = int(value.y*height)
        points[index] = (x,y)
        cv2.circle(frame, (x,y), 4, (0,255,255), -1)
    
    for start,end in HAND_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 1)
    
    return points

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)
timestamp_ms = 0

prev_x = None
prev_y = None

while(True):
    ret,frame = cap.read()


    if not ret:
        print("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    
    #region clear button code begins
    # Clear button coordinates 
    button_width = 200
    button_height = 70

    # Left-middle position
    x1 = 20
    y1 = (height - button_height) // 2

    x2 = x1 + button_width
    y2 = y1 + button_height

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Write CLEAR inside it
    text = "CLEAR"

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2
    )

    text_x = x1 + (button_width - text_width) // 2
    text_y = y1 + (button_height + text_height) // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        5
    )

    #endregion


    #region draw button code begins

    margin = 20

    x1_top = x1
    y1_top = y1 - button_height - margin

    x2_top = x1_top + button_width
    y2_top = y1_top + button_height

    cv2.rectangle(frame, (x1_top, y1_top), (x2_top, y2_top), (255, 0, 0), 2)
    cv2.putText(frame, "SAVE", (x1_top + 45, y1_top + 45),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    #endregion


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms += 1
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Draw skeleton
        points = draw_landmark_and_connect(frame, landmarks, width, height)

        # Index finger tip = landmark 8
        x, y = points[8]
        x_thumb, y_thumb = points[4]

        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

        if x1 <= x <= x2 and y1 <= y <= y2:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        distance = math.sqrt((x - x_thumb)**2 + (y - y_thumb)**2)

        if distance < 60:
            if prev_x is None:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 255), 5)
            prev_x, prev_y = x, y

        else:
            prev_x = None
            prev_y = None


    output = cv2.add(frame, canvas)
    cv2.imshow("AirScribble", output)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
landmarker.close()

