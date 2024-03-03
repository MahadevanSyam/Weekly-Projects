import cv2
import mediapipe as mp
import numpy as np
from collections import deque

#deque for storing the points were it is drawn
draw_points = [deque(maxlen=1024)]
draw_index = 0

# mediapipe functions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence = 0.6)
mp_landmarks = mp.solutions.drawing_utils

# webcam
cap = cv2.VideoCapture(0)
data, frame_old= cap.read()
while data:
    data, frame = cap.read()
    frame = cv2.resize(frame,(800,600))
    frame = cv2.flip(frame,1)
    frame_color = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # clear and draw button
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (0, 0, 0), 2)
    cv2.putText(frame, "CLEAR", (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "DRAW", (175, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # hand detection
    results = hands.process(frame_color)

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                x_point = int(point.x * 800)
                y_point = int(point.y * 600)
                landmarks.append([x_point,y_point])

            # drawing on frame
            mp_landmarks.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        index = (landmarks[8][0], landmarks[8][1])
        center = index
        middle = (landmarks[12][0], landmarks[12][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        if middle[1] - center[1] < 30:
            draw_points.append(deque(maxlen=512))
            draw_index += 1
        elif center[1] <= 65:
            if 40 <= center[0] <= 140:
                draw_points = [deque(maxlen=512)]
                draw_index = 0
        else:
            draw_points[draw_index].appendleft(center)
    else:
        draw_points.append(deque(maxlen=512))
        draw_index += 1

    points = [draw_points]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], [0,0,0] , 2)
    cv2.imshow("SPS Project",frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
