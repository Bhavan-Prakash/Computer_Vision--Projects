import numpy as np
import cv2
import mediapipe as mp


from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# curl -o hand_landmarker.task \
#  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

MODEL_PATH = "/Users/bhavan/Documents/codes/Computer_Vision/Air-Board/hand_landmarker.task"

base_options = mp_python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

prev_x = None
prev_y = None

res, frame = cap.read()
height, width, _ = frame.shape

canvas = np.zeros((height, width, 3), dtype=np.uint8)

while(True):
    res, frame = cap.read()

    if not res:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #region clear button code begins
    # Clear button coordinates 
    button_width = 200
    button_height = 70

    # Left-middle position
    x1_clear_button = 20
    y1_clear_button = (height - button_height) // 2

    x2_clear_button = x1_clear_button + button_width
    y2_clear_button = y1_clear_button + button_height

    # Draw rectangle
    cv2.rectangle(frame, (x1_clear_button, y1_clear_button), (x2_clear_button, y2_clear_button), (0, 0, 255), 2)

    # Write CLEAR inside it
    text = "CLEAR"

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2
    )

    text_x = x1_clear_button + (button_width - text_width) // 2
    text_y = y1_clear_button + (button_height + text_height) // 2

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

    x1_top = x1_clear_button
    y1_top = y1_clear_button - button_height - margin

    x2_top = x1_top + button_width
    y2_top = y1_top + button_height

    cv2.rectangle(frame, (x1_top, y1_top), (x2_top, y2_top), (255, 0, 0), 2)
    cv2.putText(frame, "SAVE", (x1_top + 45, y1_top + 45),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    #endregion

    height, width, _ = frame.shape

    mp_image = mp.Image(
        image_format= mp.ImageFormat.SRGB,
        data = rgb_frame
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hands in result.hand_landmarks:
            for i, coordinate in enumerate(hands):
                px = int(coordinate.x * width)
                py = int(coordinate.y * height)
                if i==8:
                    cv2.circle(frame, (px,py), 5, (0,255,0), -1)

                    if x1_clear_button < px < x2_clear_button and y1_clear_button < py < y2_clear_button:
                        canvas[:] = 0

                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x,prev_y), (px,py), (0,255,0), 5)
                    prev_x = px
                    prev_y = py
    else:
        # Finger lost -> stop connecting old point to new point
        prev_x = None
        prev_y = None

    # Overlay drawing on camera feed
    output = cv2.add(frame, canvas)

    cv2.imshow("testing", output)

    # press esc key to close
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()