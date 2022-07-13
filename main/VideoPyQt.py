from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QLibraryInfo
import numpy as np
import mediapipe as mp
import os
import csv
from keras.models import load_model

"Fix version of Qt with which opencv was compiled is not similar to the one used by PyQt5 causing a conflict."
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()

        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_holistic = mp.solutions.holistic
        
        self.__config()
        
        self._run_flag = True
        
    def __config(self):
        # self.DATA_PATH = os.path.join('dataset')

        # self.model = load_model("model-cnn-facerecognition.h5")
        # self.actions = os.listdir(os.path.join(self.DATA_PATH))
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),
                       (245, 17, 116), (17, 245, 196), (116, 17, 245)]

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
        ) if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60+num*40),
                        (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame

    def run(self, videoSource=0):
        cap = cv2.VideoCapture(videoSource)
        sequence = []
        if not cap.isOpened():
            raise ValueError("Galat membuka sumber video", videoSource)
        else:
            with self.__mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as Holistic:
                while self._run_flag:
                    ret, cv_img = cap.read()

                    if ret:
                        image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        results = Holistic.process(image)

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
                        self.change_pixmap_signal.emit(image)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class DatasetCapturer(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, letter, totalSequence):
        super().__init__()

        self.letter = letter
        self.totalSequence = totalSequence
        
        self.DATA_PATH = os.path.join('dataset')

        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_holistic = mp.solutions.holistic

        self.__autoCreateDir()
        
        self.lastNum = self.__newestDir()
        
        self._run_flag = True
    
    def __autoCreateDir(self):
        if not os.path.isdir(os.path.join(self.DATA_PATH)):
            os.mkdir(self.DATA_PATH)
        if not os.path.isdir(os.path.join(self.DATA_PATH, self.letter)):
            os.mkdir(os.path.join(self.DATA_PATH, self.letter))
            
        os.mkdir(os.path.join(self.DATA_PATH, self.letter,
                              str(self.__newestDir() + 1)))

    def __newestFile(self):
        listFile = os.listdir(os.path.join(self.DATA_PATH, self.letter))

        if listFile == []:
            return 0

        return max([int(f[:f.index('.')])
                    for f in listFile])
        
    def __newestDir(self):
        listFile = os.listdir(os.path.join(self.DATA_PATH, self.letter))

        if listFile == []:
            return 0

        return max([int(f) for f in listFile])

    def createModelHeader(self):
        landmarks = ['class']
        for val in range(1, 34):
            landmarks += ['x{}'.format(val), 'y{}'.format(val),
                          'z{}'.format(val), 'v{}'.format(val)]

        with open(self.filePath, mode='w', newline='') as f:
            csv_writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
        ) if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def run(self, videoSource=0):
        cap = cv2.VideoCapture(videoSource)
        
        currentSeq = 0
        
        if not cap.isOpened():
            raise ValueError("Galat membuka sumber video", videoSource)
        else:
            with self.__mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as Holistic:
                while self._run_flag:
                    ret, cv_img = cap.read()

                    if ret:
                        image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        results = Holistic.process(image)

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

                        if results.pose_landmarks:
                            if currentSeq < self.totalSequence:
                                keypoints = self.extract_keypoints(
                                    results) 
                                npy_path = os.path.join(
                                    self.DATA_PATH, self.letter, str(self.lastNum), str(currentSeq) + '.npy')
                                np.save(npy_path, keypoints)
                                
                                currentSeq += 1
                            else:
                                break
                                
                        self.change_pixmap_signal.emit(image)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setupUi()

        self.__init_capture_time()

        self.thread = VideoThread()

        self.webcamView()

    def setupUi(self):
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 640
        self.display_height = 480
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 510, 801, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(
            self.horizontalLayoutWidget)
        self.horizontalLayout.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(10, 0, 10, 0)
        self.horizontalLayout.setSpacing(20)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.addDatasetBtn()
        self.statusText()
        self.labelText()
        self.comboBox()

        self.horizontalLayout.setStretch(1, 5)
        self.horizontalLayout.setStretch(3, 1)

        self.retranslateUi()
        self.alfabertComBox.setCurrentIndex(0)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.horizontalLayoutWidget)
        self.setLayout(vbox)

    def addDatasetBtn(self):
        self.buttonAddDataset = QtWidgets.QPushButton(
            self.horizontalLayoutWidget)
        self.buttonAddDataset.clicked.connect(self.addDataset)
        self.buttonAddDataset.setObjectName("buttonAddDataset")
        self.horizontalLayout.addWidget(self.buttonAddDataset)

    def statusText(self):
        self.statusText = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.statusText.setAlignment(QtCore.Qt.AlignCenter)
        self.statusText.setObjectName("statusText")
        self.horizontalLayout.addWidget(self.statusText)

    def labelText(self):
        self.hurufText = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.hurufText.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.hurufText.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.hurufText.setObjectName("hurufText")
        self.horizontalLayout.addWidget(self.hurufText)

    def comboBox(self):
        self.alfabertComBox = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.alfabertComBox.setMaxCount(26)
        self.alfabertComBox.setObjectName("alfabertComBox")
        self.alfabertComBox.setEditable(True)
        self.alfabertComBox.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.alfabertComBox.lineEdit().setReadOnly(True)
        self.horizontalLayout.addWidget(self.alfabertComBox)

    def retranslateUi(self):
        self.buttonAddDataset.setText("Tambah dataset")
        self.statusText.setText("")
        self.hurufText.setText("Huruf :")
        self.alfabertComBox.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        self.alfabertComBox.currentIndexChanged.connect(self.selectedLetter)

    def timer_start(self):
        self.timers = QtCore.QTimer(self)
        self.timers.timeout.connect(self.timer_timeout)
        self.timers.start(1000)

    def timer_timeout(self):
        if self.waitNotification > 0:
            self.waitNotification -= 1
            self.statusText.setText(
                "Mengambil sampel dalam " + str(self.waitNotification) + " detik")

            if self.waitNotification == 0:
                self.captureDataset()

        elif self.thread.isRunning():
            self.statusText.setText("Sedang mengambil sampel...")
            
        else:
            self.webcamView()
            self.__init_capture_time()
            self.timers.stop()

    def __init_capture_time(self):
        self.totalSequence = 30
        self.waitNotification = 5
        self.statusText.setText("")

    def webcamView(self):
        if self.thread is not VideoThread:
            self.thread.stop()

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def captureDataset(self):
        if self.thread is not DatasetCapturer:
            self.thread.stop()

        self.thread = DatasetCapturer(
            letter=self.alfabertComBox.itemText(self.alfabertComBox.currentIndex()), totalSequence=self.totalSequence)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def selectedLetter(self, index):
        print(self.alfabertComBox.itemText(self.alfabertComBox.currentIndex()))

    @pyqtSlot()
    def addDataset(self):
        self.timer_start()

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
            self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
