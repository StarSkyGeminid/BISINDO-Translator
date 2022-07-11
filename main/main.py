import cv2
import numpy as np
import pickle
import mediapipe as mp
import pandas as pd
import csv
import os


class Main:
    """Class utama yang berfungsi untuk menampilkan GUI"""

    def __init__(self, title):
        self.__config()
        self.title = title

    def __config(self):
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_holistic = mp.solutions.holistic
        self.__mp_drawing_styles = mp.solutions.drawing_styles

        with open('./dataset/model.pkl', 'rb') as f:
            self.__model = pickle.load(f)

    def start(self, videoSource=0):
        self.cam = cv2.VideoCapture(videoSource)
        self.__createWindow()

    def __createWindow(self):
        with self.__mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
            while self.cam.isOpened():
                ret, frame = self.cam.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.__mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS,
                                                 landmark_drawing_spec=self.__mp_drawing_styles
                                                 .get_default_hand_landmarks_style(),
                                                 )

                self.__mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS,
                                                 landmark_drawing_spec=self.__mp_drawing_styles
                                                 .get_default_hand_landmarks_style(),
                                                 )
                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    row = pose_row

                    X = pd.DataFrame([row])
                    body_language_class = self.__model.predict(X)[0]
                    body_language_prob = self.__model.predict_proba(X)[0]

                    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                    cv2.putText(image, 'CLASS', (95, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[
                                0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'PROB', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (
                        10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

                cv2.imshow(self.title, image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    break

    def __del__(self):
        if self.cam.isOpened():
            self.cam.release()
