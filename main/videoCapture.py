import cv2
import numpy as np
import pickle
import mediapipe as mp
import pandas as pd
import csv
import os

class FaceRecognition:
    """Class yang berfungsi untuk mengambil gambar dari kamera serta memproses nya dengan dataset yang berasal dari model.pkl"""

    def __init__(self, video_source=0, sizePercent=75):
        self.vid = cv2.VideoCapture(video_source)

        self.sizePercent = sizePercent
        # Open the video source
        if not self.vid.isOpened():
            raise ValueError("Galat membuka sumber video", video_source)

        self.__config()

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * sizePercent / 100
        self.height = self.vid.get(
            cv2.CAP_PROP_FRAME_HEIGHT) * sizePercent / 100

    def __config(self):
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_holistic = mp.solutions.holistic

        with open('./model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def get_frame(self, save=False, text='A'):
        if save and not os.path.exists('dataset.csv'):
            landmarks = ['class']
            for val in range(1, 34):
                landmarks += ['x{}'.format(val), 'y{}'.format(val),
                            'z{}'.format(val), 'v{}'.format(val)]
            with open('dataset.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)
                
        with self.__mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
            if self.vid.isOpened():
                ret, frame = self.vid.read()
                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    results = holistic.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    self.__mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS,
                                                     self.__mp_drawing.DrawingSpec(
                                                         color=(80, 22, 10), thickness=2, circle_radius=4),
                                                     self.__mp_drawing.DrawingSpec(
                                                         color=(80, 44, 121), thickness=2, circle_radius=2)
                                                     )

                    self.__mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS,
                                                     self.__mp_drawing.DrawingSpec(
                                                         color=(121, 22, 76), thickness=2, circle_radius=4),
                                                     self.__mp_drawing.DrawingSpec(
                                                         color=(121, 44, 250), thickness=2, circle_radius=2)
                                                     )

                    try:
                        # Mengambil Pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array(
                            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                        row = pose_row

                        if not save:
                            X = pd.DataFrame([row])
                            body_language_class = self.model.predict(X)[0]
                            body_language_prob = self.model.predict_proba(X)[0]
                            print(body_language_class, body_language_prob)

                            cv2.rectangle(image, (0, 0), (250, 60),
                                        (245, 117, 16), -1)

                            cv2.putText(image, 'Huruf', (95, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image, body_language_class.split(' ')[
                                        0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                            cv2.putText(image, 'Prob', (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (
                                10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        else:
                            row = pose_row

                            row.insert(0, text)

                            with open('dataset.csv', mode='a', newline='') as f:
                                csv_writer = csv.writer(
                                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(row)
                    except:
                        pass

                    return (ret, self.rescale_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.sizePercent))

        return (ret, None)

    def rescale_frame(self, frame, sizePercent=75):
        width = int(frame.shape[1] * sizePercent / 100)
        height = int(frame.shape[0] * sizePercent / 100)
        dim = (width, height)

        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def stop(self):
        if self.vid.isOpened():
            self.vid.release()
            cv2.destroyAllWindows()
            print("Video source ditutup")
            return True

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
