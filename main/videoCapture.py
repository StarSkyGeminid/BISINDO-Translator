import cv2
from keras.models import load_model
import numpy as np
import pickle
import mediapipe as mp
import pandas as pd

class FaceRecognition:
    def __init__(self, video_source=0, sizePercent=75):
        self.vid = cv2.VideoCapture(video_source)
        
        self.sizePercent = sizePercent
        # Open the video source
        if not self.vid.isOpened():
            raise ValueError("Galat membuka sumber video", video_source)

        self.__config()

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * sizePercent/ 100
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * sizePercent/ 100
 
    def __config(self):
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_holistic = mp.solutions.holistic
        
        with open('./model.pkl', 'rb') as f:
            self.model = pickle.load(f)
 
    def get_frame(self):
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

                    # 3. Left Hand
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

                        X = pd.DataFrame([row])
                        body_language_class = self.model.predict(X)[0]
                        body_language_prob = self.model.predict_proba(X)[0]
                        print(body_language_class, body_language_prob)

                        # Ambil koordinat telinga
                        coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[self.__mp_holistic.PoseLandmark.LEFT_EAR].x,
                                results.pose_landmarks.landmark[self.__mp_holistic.PoseLandmark.LEFT_EAR].y)), [640, 480]).astype(int))

                        cv2.rectangle(image,
                                    (coords[0], coords[1]+5),
                                    (coords[0]+len(body_language_class)
                                    * 20, coords[1]-30),
                                    (245, 117, 16), -1)
                        cv2.putText(image, body_language_class, coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # status box
                        cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                        # Display Class
                        cv2.putText(image, 'CLASS', (95, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, body_language_class.split(' ')[
                                    0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Display prob
                        cv2.putText(image, 'PROB', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (
                            10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    except:
                        pass
                    
                    return (ret, self.rescale_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.sizePercent))
                else:
                    return (ret, None)
            else:
                return (ret, None)
 
    def rescale_frame(self, frame, sizePercent=75):
        width = int(frame.shape[1] * sizePercent/ 100)
        height = int(frame.shape[0] * sizePercent/ 100)
        dim = (width, height)
        
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
