# https://www.youtube.com/watch?v=pG4sUNDOZFg

import cv2
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
import mediapipe as mp
import time
from matplotlib import pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)  # Number is to find the good webcam

LILA = (243, 79, 124)
LIGHT_LILA = (248, 151, 177)
GREEN = (177, 248, 151)

# Initiate holistic Model
black = np.zeros([20, 20, 1], dtype="uint8")
white = np.full((500, 500, 3), 255, dtype=np.uint8)
drawer = cv2.cvtColor(black, cv2.COLOR_RGB2BGR)
drawer = cv2.resize(drawer, (1600, 900))
drawer_shoulders = cv2.cvtColor(black, cv2.COLOR_RGB2BGR)
drawer_shoulders = cv2.resize(drawer, (1600, 900))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    session = 0
    while (cap.isOpened()):
        sequence = 0
        while (sequence < 5):

            picture_save = 'picture' + str(session) + '_' + str(sequence) + '.png'
            time_end = time.time() + 10
            while (time.time() < time_end):
                ret, frame = cap.read()

                # recolor feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Make detection
                results = holistic.process(image)

                # Right Hand
                connections_wanted = frozenset(
                    [(16, 18), (14, 16), (12, 14), (11, 13), (13, 15), (15, 17), (24, 26), (26, 28), (28, 30),
                     (30, 32), (23, 25), (25, 27), (27, 29), (29, 31)])
                connection_shoulders = frozenset([(11, 12)])
                # Right Hand
                mp_drawing.draw_landmarks(
                    image=drawer,
                    landmark_list=results.right_hand_landmarks,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=LIGHT_LILA,
                        thickness=1,
                        circle_radius=0,
                    )
                )
                # Left Hand
                mp_drawing.draw_landmarks(
                    image=drawer,
                    landmark_list=results.left_hand_landmarks,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=LIGHT_LILA,
                        thickness=0,
                        circle_radius=0,
                    )
                )
                # Pose detection
                mp_drawing.draw_landmarks(
                    image=drawer,
                    landmark_list=results.pose_landmarks,
                    connections=mp_holistic.POSE_CONNECTIONS.intersection(connections_wanted),
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=LIGHT_LILA,
                        thickness=0,
                        circle_radius=0,
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=LILA,
                        thickness=1,
                        circle_radius=0,
                    )
                )
                mp_drawing.draw_landmarks(
                    image=drawer_shoulders,
                    landmark_list=results.pose_landmarks,
                    connections=mp_holistic.POSE_CONNECTIONS.intersection(connection_shoulders),
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=LIGHT_LILA,
                        thickness=0,
                        circle_radius=0,
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=LILA,
                        thickness=1,
                        circle_radius=0,
                    )
                )
                drawer_shoulders = cv2.GaussianBlur(drawer_shoulders, (5, 5), 0)
                drawer = cv2.addWeighted(drawer, 1, drawer_shoulders, 0.1, 0)
                cv2.imshow('drawer', drawer)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            time.sleep(5)
            cv2.imwrite(picture_save, drawer)
            drawer = cv2.imread('black.png')
            drawer_shoulders = cv2.imread('black.png')
          #  drawer = cv2.resize(drawer, (1600, 900))
            sequence += 1
        session += 1
        # if sequence == 5:
        # STOCKER SOMME DES 5 SEQUENCES

cap.release()
cv2.destroyAllWindows()
