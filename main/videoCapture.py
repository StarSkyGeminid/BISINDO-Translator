import cv2
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

class FaceRecognition:
    def __init__(self, video_source=0, sizePercent=75):
        self.vid = cv2.VideoCapture(video_source)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml');

        self.model = load_model("model-cnn-facerecognition.h5")

        self.sizePercent = sizePercent
        # Open the video source
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.__config()
        
          # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) * sizePercent/ 100
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * sizePercent/ 100
 
    def __config(self):
        dataset = self.generate_dataset(500)
        Y = dataset[:, 2]
        
        le = LabelEncoder()
    
        le.fit(Y)
        
        self.labels = le.classes_
        print(self.labels)

 
    def generate_dataset(self, size, classes=5, noise=10.5):
        # Generate random datapoints
        labels = np.random.randint(0, classes, size)
        x1 = (np.random.rand(size) + labels) / classes
        x2 = x1**2 + np.random.rand(size) * noise
        
        # Reshape data in order to merge them
        x1 = x1.reshape(size, 1)
        x2 = x2.reshape(size, 1)
        labels = labels.reshape(size, 1)
    
        # Merge the data
        data = np.hstack((x1, x2, labels))
        return data
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h) in faces:
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (50, 50))
                    face_img = face_img.reshape(1, 50, 50, 1)

                    result = self.model.predict(face_img)
                    idx = result.argmax(axis=1)
                    confidence = result.max(axis=1)*100
                    
                    if confidence > 90:
                        label_text = "%s (%.2f %%)" % (self.labels[idx], confidence)
                    else:
                        label_text = "N/A"
                    frame = self.__draw_ped(frame, label_text, x, y, x + w, y + h,
                                     color=(0, 255, 255), text_color=(50, 50, 50))
                # return (ret, self.rescale_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.sizePercent))
                return (ret, self.rescale_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.sizePercent))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
    def rescale_frame(self, frame, sizePercent=75):
        width = int(frame.shape[1] * sizePercent/ 100)
        height = int(frame.shape[0] * sizePercent/ 100)
        dim = (width, height)
        
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    
    def __draw_ped(self, img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img,
                    (x0, y0 + baseline),  
                    (max(xt, x0 + w), yt), 
                    color, 
                    2)
        cv2.rectangle(img,
                    (x0, y0 - h),  
                    (x0 + w, y0 + baseline), 
                    color, 
                    -1)  
        cv2.putText(img, 
                    label, 
                    (x0, y0),                   
                    cv2.FONT_HERSHEY_SIMPLEX,     
                    0.5,                          
                    text_color,                
                    1,
                    cv2.LINE_AA) 
        return img
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
