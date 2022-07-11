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

"Fix version of Qt with which opencv was compiled is not similar to the one used by PyQt5 causing a conflict."
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_hands = mp.solutions.hands

        self._run_flag = True

    def run(self, videoSource=0):
        cap = cv2.VideoCapture(videoSource)

        if not cap.isOpened():
            raise ValueError("Galat membuka sumber video", videoSource)
        else:
            with self._mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
                while self._run_flag:
                    ret, cv_img = cap.read()

                    if ret:
                        image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        results = hands.process(image)

                        image.flags.writeable = True
                        image = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

                        if results.multi_hand_landmarks:
                            for hand in results.multi_hand_landmarks:
                                self._mp_drawing.draw_landmarks(
                                    cv_img, hand, self._mp_hands.HAND_CONNECTIONS)
                        if ret:
                            self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class DatasetCapturer(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, letter):
        super().__init__()

        self.letter = letter


        self.DATA_PATH = os.path.join('dataset')
        
        self.__config()

        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_holistic = mp.solutions.holistic

        self._run_flag = True

    def __config(self):
        self.filePath = 'dataset.csv'

        if not os.path.exists(self.filePath):
            self.createFirstModel()

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
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    def run(self, videoSource=0):
        cap = cv2.VideoCapture(videoSource)

        if not cap.isOpened():
            raise ValueError("Galat membuka sumber video", videoSource)
        else:
            with self.__mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as Holistic:
                while self._run_flag:
                    ret, cv_img = cap.read()

                    image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    results = Holistic.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

                    self.__mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS,
                                                     landmark_drawing_spec=self.__mp_drawing_styles
                                                     .get_default_hand_landmarks_style(),
                                                     )

                    self.__mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.__mp_holistic.HAND_CONNECTIONS,
                                                     landmark_drawing_spec=self.__mp_drawing_styles
                                                     .get_default_hand_landmarks_style(),
                                                     )

                    if results.pose_landmarks:
                        keypoints = self.extract_keypoints(results)  # NEW Export keypoints
                        npy_path = os.path.join(self.DATA_PATH, self.letter)
                        np.save(npy_path, keypoints)

                    if ret:
                        self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setupUi()

        self.waitNotification = 5
        self.captureTime = 5

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

        elif self.captureTime > 0:
            self.captureTime -= 1
            self.statusText.setText("Sedang mengambil sampel...")

            if self.captureTime == 0:
                self.webcamView()
                self.captureTime = 5
                self.waitNotification = 5
                self.statusText.setText("")
                self.timers.stop()

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
            letter=self.alfabertComBox.itemText(self.alfabertComBox.currentIndex()))
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def selectedLetter(self, index):
        print(self.alfabertComBox.itemText(index))

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
