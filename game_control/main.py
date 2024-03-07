import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Constants
dw = 640
dh = 480

# mediapipe functions
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence = 0.6)
mp_landmarks = mp.solutions.drawing_utils

# initialise webcam
cap = cv2.VideoCapture(0)
data, frame = cap.read()
while data:
    data, frame = cap.read()
    frame = cv2.resize(frame, (dw, dh))
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # hand detection
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_landmarks.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            middle_finger_landmarks = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            middle_finger_main = middle_finger_landmarks.y

            threshold = 0.7

            if middle_finger_main < threshold:
                cv2.putText(frame, "Middle Finger Up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.keyDown('right')
                pyautogui.keyUp('left')
            else:
                cv2.putText(frame, "Middle Finger Down", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.keyDown('left')
                pyautogui.keyUp('right')

    cv2.imshow("SPS Project(Developed By Mahadevan Syam)", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
