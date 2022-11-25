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
white = np.full((500, 500, 3), 255, dtype=np.uint8)
drawer = cv2.cvtColor(white, cv2.COLOR_RGB2BGR)
drawer = cv2.resize(drawer, (1200, 1000))
i = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while (cap.isOpened()):
        sequence = 0
        while (sequence < 5):
            
            time_end = time.time() + 10
            while (time.time() < time_end):
                
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

                #Right Hand
                mp_drawing.draw_landmarks(
                    image=drawer,
                    landmark_list=results.right_hand_landmarks,
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=-50,
                        circle_radius=0,
                    )
                    )
                #Left Hand
                mp_drawing.draw_landmarks(
                    image=drawer,
                    landmark_list=results.left_hand_landmarks,
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=-50,
                        circle_radius=0,
                    )
                    )
                #Pose detection
                mp_drawing.draw_landmarks(
                    image=drawer,
                    landmark_list=results.pose_landmarks,
                    connections=mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=-50,
                        circle_radius=0,
                    )
                    )
                cv2.imshow('Raw Webcam Feed', drawer)
                drawer = cv2.resize(drawer, (1000, 800))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                # STOCKER LA SEQUENCE ISOLEE
            time.sleep(20)
            sequence += 1
            drawer = cv2.cvtColor(white, cv2.COLOR_RGB2BGR)
            #if sequence == 5:
                # STOCKER SOMME DES 5 SEQUENCES
        #i += 1

cap.release()
cv2.destroyAllWindows()