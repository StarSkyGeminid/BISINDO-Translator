from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import mediapipe as mp 


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._mp_drawing = mp.solutions.drawing_utils 
        self._mp_hands = mp.solutions.hands 
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        with self._mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
            while self._run_flag:
                ret, cv_img = cap.read()

                image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False   

                results = hands.process(image)

                image.flags.writeable = True   
                image = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand in results.multi_hand_landmarks:
                        self._mp_drawing.draw_landmarks(cv_img, hand, self._mp_hands.HAND_CONNECTIONS)

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
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 640
        self.display_height = 480
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.comboBox = QComboBox()
        self.comboBox.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
        self.comboBox.currentIndexChanged.connect(self.selection_class)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.comboBox)
        self.setLayout(vbox)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def selection_class(self, i):

        for count in range(self.comboBox.count()):
            result = self.comboBox.itemText(count)
            print(result)

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
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())