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

while(True):
    res, frame = cap.read()

    if not res:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
                cv2.circle(frame, (px,py), 5, (0,255,0), -1)

    cv2.imshow("testing", frame)

    # press esc key to close
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()