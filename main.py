#https://www.youtube.com/watch?v=pG4sUNDOZFg

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import mediapipe as mp
import time
from matplotlib import pyplot as plt
#from numpy import angle
#from utils import *

def transform_range(value, old_range, new_range):
    old_min = old_range[0]
    old_max = old_range[1]
    new_min = new_range[0]
    new_max = new_range[1]
    old_range = (old_min, old_max)
    new_range = (new_min, new_max)
    old_span = old_max - old_min
    new_span = new_max - new_min
    scaled = float(value - old_min) / float(old_span) * float(new_span) + new_min
    return int(scaled)

#midiout selection

def distance(a, b):
    point1 = np.array([a.x, a.y])
    point2 = np.array([b.x, b.y])
    return np.linalg.norm(point1 - point2)

def angle(a,b,c):
    # law of cosines to angle in b
    side_a = distance(b, c)
    side_b = distance(a, c)
    side_c = distance(a, b)
    numerator = np.square(side_a)+np.square(side_c)-np.square(side_b)
    cosine_b = numerator/(2*side_a*side_c)
    angle_b = np.arccos(cosine_b)
    return math.floor(np.degrees(angle_b))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0) # Number is to find the good webcam

# Initiate holistic Model
black = np.zeros([200, 250, 1], dtype="uint8")
white = np.full((500, 500, 3), 255, dtype=np.uint8)
drawer = cv2.cvtColor(black, cv2.COLOR_RGB2BGR)
drawer = cv2.resize(drawer, (1600, 900))
cv2.resizeWindow('drawer', (1600, 900))
i = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    last_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()

        #recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Make detection
        results = holistic.process(image)
        # print(results.pose_landmarks) # Pour le corps ?
        # print(results.left_hand_landmarks)
        # print(results.right_hand_landmarks)
        #mp_holistic.HAND_CONNECTIONS = lancer cette ligne pour savoir ce qu'il se passe. On peut mettre des if ou des = à qui déclenche tel ou tel truc
        # print(results.face_landmarks)

        # Recolor image back to BGR for rendering

        #draw face landmarks
        #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

        connections_wanted = frozenset([(16, 18), (14, 16), (12, 14), (11, 12), (11, 13), (13, 15), (15, 17), (24, 26), (26, 28), (28, 30), (30, 32), (23, 25), (25, 27), (27, 29), (29, 31)])
        #Right Hand
        mp_drawing.draw_landmarks(
            image=drawer,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS.intersection(connections_wanted),
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=0,
                circle_radius=0,
            )
        )
        #Left Hand
        mp_drawing.draw_landmarks(
            image=drawer,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS.intersection(connections_wanted),
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=0,
                circle_radius=0,
            )
        )
        #Pose detection
        mp_drawing.draw_landmarks(
            image=drawer,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS.intersection(connections_wanted),
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),
                thickness=0,
                circle_radius=0,
            )
        )
        print(mp_holistic.POSE_CONNECTIONS)
        if (time.time() - last_time) > 0.2:
            if results.pose_landmarks != None:
                #Positions membres

                x_left_hand = results.pose_landmarks.landmark[15].x
                x_right_hand = results.pose_landmarks.landmark[16].x
                y_left_hand = results.pose_landmarks.landmark[15].y
                y_right_hand = results.pose_landmarks.landmark[16].y
                #PIEDS
                x_left_feet = results.pose_landmarks.landmark[32].x
                x_right_feet = results.pose_landmarks.landmark[31].x
                #NEZ
                x_nose = results.pose_landmarks.landmark[0].x
                y_nose = results.pose_landmarks.landmark[0].y
                #GENOUX
                x_left_knee = results.pose_landmarks.landmark[25].x
                x_right_knee = results.pose_landmarks.landmark[26].x
                #INDEX_HAND
                x_left_index = results.pose_landmarks.landmark[19].x
                x_right_index = results.pose_landmarks.landmark[20].x

                right_shoulder = results.pose_landmarks.landmark[12]
                right_elbow = results.pose_landmarks.landmark[14]
                right_wrist = results.pose_landmarks.landmark[16]
                left_shoulder = results.pose_landmarks.landmark[11]
                left_elbow = results.pose_landmarks.landmark[13]
                left_wrist = results.pose_landmarks.landmark[15]

                #Calculs
                # absolute value of the rest of left hand and right hand
                hand_abs = abs(x_left_hand - x_right_hand)
                hand_abs_y = abs(y_left_hand - y_right_hand)
                #DIFFERENCE_PIEDS_MAINS
                hand_knee_left_abs_x = abs(x_left_knee - x_left_index)
                hand_knee_right_abs_x = abs(x_right_knee - x_right_index)

                feet_abs = abs(x_left_feet - x_right_feet)

                angle_2d_right_arm = angle(right_shoulder, right_elbow, right_wrist)
                angle_2d_left_arm = angle(left_shoulder, left_elbow, left_wrist)

                #print(f'2D: {angle_2d_right_arm:.2f}')
                #DISTANCE PIEDS
                #send_note(transform_range(feet_abs, (0, 1.1), (49, 71)), transform_range(y_nose, (0, 1.1), (25, 127)), 0)
                #MAIN DROITE HAUTEUR
                #send_note(transform_range(y_right_hand, (0, 1.1), (49, 71)), transform_range(y_nose, (0, 1.1), (25, 127)), 0)
                #ANGLE BRAS DROIT
                #DIFFERENCE MAINS
                #send_note(transform_range(hand_abs, (0, 1.1), (69, 104)), transform_range(hand_abs_y, (0, 1.1), (25, 127)), 1)
                #ANGLE_BRAS_GAUCHE
                #print(angle_2d_right_arm)
                #send_note(transform_range(angle_2d_right_arm, (0, 1000), (69, 127)), 100, 1)
                #DIFFERENCE_GENOUX_MAINS
                #send_note(transform_range(hand_knee_left_abs_x, (0, 1.1), (69, 104)), transform_range(hand_knee_right_abs_x, (0, 1.1), (45, 127)), 1)


            last_time = time.time()
        cv2.imshow('Raw Webcam Feed', drawer)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()