from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QComboBox, QTableWidgetItem
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QLibraryInfo
import numpy as np
import os
import csv
from keras.models import load_model
import datetime
import csv
from itertools import zip_longest

"Fix version of Qt with which opencv was compiled is not similar to the one used by PyQt5 causing a conflict."
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    recognizable_face = pyqtSignal(str)

    def __init__(self, videoSource=0):
        super().__init__()
        
        self.videoSource = videoSource
        self.__config()

        self._run_flag = True

    def __config(self):
        # self.DATA_PATH = os.path.join('dataset')

        self.model = load_model("output/trained-model.h5")
        self.label = np.load('output/labels.npy')
        self.face_cascade = cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')

        # self.actions = os.listdir(os.path.join(self.DATA_PATH))
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),
                       (245, 17, 116), (17, 245, 196), (116, 17, 245)]

    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60+num*40),
                          (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame

    def draw_ped(self, img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):
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

    def run(self):
        cap = cv2.VideoCapture(self.videoSource)
        if not cap.isOpened():
            raise ValueError("Galat membuka sumber video", self.videoSource)
        else:
                while self._run_flag:
                    ret, frame = cap.read()
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
                                label_text = "%s (%.2f %%)" % (
                                    self.label[idx], confidence)
                                self.recognizable_face.emit(str(self.label[idx][0]))
                            else :
                                label_text = "N/A"
                                
                            frame = self.draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))
                        self.change_pixmap_signal.emit(frame)

                    else :
                        break
                
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class FaceDetection(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, videoSource=0):
        super().__init__()
        
        self.videoSource = videoSource
        self.__config()

        self._run_flag = True

    def __config(self):
        # self.DATA_PATH = os.path.join('dataset')

        # self.model = load_model("model-cnn-facerecognition.h5")
        # self.actions = os.listdir(os.path.join(self.DATA_PATH))
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),
                       (245, 17, 116), (17, 245, 196), (116, 17, 245)]

    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60+num*40),
                          (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame

    def run(self):
        cap = cv2.VideoCapture(self.videoSource)
        if not cap.isOpened():
            raise ValueError("Galat membuka sumber video", self.videoSource)
        else:
                while self._run_flag:
                    ret, cv_img = cap.read()

                    if ret:
                        image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False


                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        self.change_pixmap_signal.emit(image)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class App(QWidget):
    def __init__(self, title):
        super().__init__()
        
        self.title = title
        self.setupUi()
        self.listName = []
        self.listTime = []
        
    def setupUi(self):
        self.setWindowTitle("Qt live label demo")

        screen = QApplication.primaryScreen()
        self.size = screen.size()
        
        
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, self.size.width(), self.size.height()))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 10, 0, 10)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.image_label = QLabel(self)
        self.image_label.resize(int(self.size.width() / 3), int(self.size.height() / 2))
        
        self.verticalLayout.addWidget(self.image_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.startBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.startBtn.setMinimumSize(QtCore.QSize(100, 40))
        self.startBtn.setObjectName("startBtn")
        self.startBtn.clicked.connect(self.startPresensi)

        self.horizontalLayout.addWidget(self.startBtn)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        
        self.stopBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.stopBtn.setMinimumSize(QtCore.QSize(100, 40))
        self.stopBtn.setObjectName("stopBtn")
        self.stopBtn.clicked.connect(self.stopPresensi)

        self.horizontalLayout.addWidget(self.stopBtn)
        
        self.downloadBtn = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.downloadBtn.setMinimumSize(QtCore.QSize(100, 40))
        self.downloadBtn.setObjectName("downloadBtn")
        self.downloadBtn.clicked.connect(self.downloadData)

        self.horizontalLayout.addWidget(self.downloadBtn)
        
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        
        self.ListPresensi = QtWidgets.QTableWidget(
            self.horizontalLayoutWidget_2)
        self.ListPresensi.setObjectName("ListPresensi")
        self.horizontalLayout_2.addWidget(self.ListPresensi)

        self.retranslateUi()
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.horizontalLayoutWidget_2)
        self.setLayout(vbox)
        
        self.InitTables()
        self.webcamView()
        
        
    def currentTime(self):
        timeNow = datetime.datetime.now()
        return timeNow.strftime("%H:%M:%S")
    
    def currentDate(self):
        timeNow = datetime.datetime.now()
        return timeNow.strftime("%Y-%m-%d")
        
    def webcamView(self):
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def InitTables(self):
        self.ListPresensi.verticalHeader().setVisible(False)
        self.ListPresensi.setColumnCount(3)
        self.ListPresensi.setHorizontalHeaderLabels(
            ['No.', 'Nama', 'Waktu'])

    def addPresensi(self, nama, waktu):
        row = self.ListPresensi.rowCount()
        self.ListPresensi.insertRow(row)
        self.ListPresensi.setItem(row, 0, QTableWidgetItem(str(row+1)))
        self.ListPresensi.setItem(row, 1, QTableWidgetItem(nama))
        self.ListPresensi.setItem(row, 2, QTableWidgetItem(waktu))

    def startPresensi(self):
        self.thread.recognizable_face.connect(self.recognizeFace)
    
    def stopPresensi(self):
        self.thread.recognizable_face.disconnect()
    
    def retranslateUi(self):
        self.startBtn.setText("Mulai")
        self.stopBtn.setText("Berhenti")
        self.downloadBtn.setText("Unduh Data")
        
    def downloadData(self):
        fileName = "Data-Presensi_" + self.currentDate() + ".csv"
        with open(fileName, 'w') as f:
            write = csv.writer(f)
            write.writerow(["Nama", "Waktu"])
            for x, y in zip_longest(self.listName, self.listTime):
                write.writerow([x, y])
            
            
    def recognizeFace(self, name):
        if name not in self.listName:
            time = self.currentTime()
            self.addPresensi(name, time)
            self.listName.append(name)
            self.listTime.append(time)
    
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            int(self.size.width() / 3), int(self.size.height() / 2), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App(title='Sistem Presensi')
    a.show()
    sys.exit(app.exec_())
