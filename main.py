import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import cv2
import os
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("can_detection_gui.ui", self)
        self.show()

        # Create Camera Feeds
        self.top_camera = CameraFeed()
        self.top_camera.image_update.connect(self.image_update_slot)
        self.left_camera = CameraFeed()
        self.left_camera.image_update.connect(self.image_update_slot)
        self.right_camera = CameraFeed()
        self.right_camera.image_update.connect(self.image_update_slot)

        # Connect Power Button
        self.power = False
        self.startup()
        self.on_pushButton.setFlat(True)
        self.on_pushButton.clicked.connect(self.turn_on)

    # Connect Camera to Label
    def image_update_slot(self, image):
        self.camera1.setPixmap(QPixmap.fromImage(image))
        self.camera2.setPixmap(QPixmap.fromImage(image))
        self.camera3.setPixmap(QPixmap.fromImage(image))

    def startup(self):
        self.on_pushButton.setStyleSheet("QPushButton { background-image: url(off.png); }")
        self.logo_label.setStyleSheet("QLabel { image: url(NJFCO_logo.png);}")

    # On/Off button click
    def turn_on(self):
        if self.power:
            self.on_pushButton.setStyleSheet("QPushButton { background-image: url(off.png); }")
            self.notification_label.setStyleSheet(open('rejected.css').read())
            self.notification_label.setText("REJECTED")
            self.top_camera.stop()
            self.left_camera.stop()
            self.right_camera.stop()
            self.power = False
        else:
            self.on_pushButton.setStyleSheet("QPushButton { background-image: url(on.png); }")
            self.notification_label.setStyleSheet(open('accepted.css').read())
            self.notification_label.setText("ACCEPTED")
            self.top_camera.start()
            self.left_camera.start()
            self.right_camera.start()
            self.power = True

# Camera Feeds Threads
class CameraFeed(QThread):
    image_update = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True
        capture = cv2.VideoCapture(0)

        while self.ThreadActive:
            ret, frame = capture.read()
            # Create QT compatible image if no issue
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                flipped_img = cv2.flip(img, 1)
                convert_to_QtFormat = QImage(flipped_img.data, flipped_img.shape[1], flipped_img.shape[0], QImage.Format_RGB888)
                pic = convert_to_QtFormat.scaled(306, 240, Qt.KeepAspectRatio)
                self.image_update.emit(pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()

def main():
    app = QApplication([])
    Root = MainWindow()
    app.exec()

if __name__ == '__main__':
    main()