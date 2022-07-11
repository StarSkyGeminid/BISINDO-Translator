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
        self.filePath = 'dataset.csv'

        if not self.vid.isOpened():
            raise ValueError("Galat membuka sumber video", video_source)
        
        if not os.path.exists(self.filePath):
            self.createFirstModel()
            
        self.model = None
        
        self.__config()

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * sizePercent / 100
        self.height = self.vid.get(
            cv2.CAP_PROP_FRAME_HEIGHT) * sizePercent / 100

    def __config(self):
        
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_holistic = mp.solutions.holistic

    def createFirstModel(self):
        landmarks = ['class']
        for val in range(1, 34):
            landmarks += ['x{}'.format(val), 'y{}'.format(val),
                        'z{}'.format(val), 'v{}'.format(val)]
            
        with open(self.filePath, mode='w', newline='') as f:
            csv_writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

    def get_frame(self, save=False, text='A'):     
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

                        if save and pose[0].visibility > 0.5:
                            row = pose_row

                            row.insert(0, text)

                            with open(self.filePath, mode='a', newline='') as f:
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
            return True

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
