import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import cv2
from cvlib.object_detection import YOLO
import csv
import datetime

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("can_detection_gui.ui", self)
        self.show()

        """ Create Camera Feeds and Connect Labels """
        self.top_camera = CameraFeed('top-detection_best.weights', 'top-detection.cfg', 'top-detection.names', 0, 'top')
        self.top_camera.image_update.connect(self.image_update_slot1)
        self.top_camera.notification_update.connect(self.notification_banner_update)
        self.left_camera = CameraFeed('sideview-yolov4-tiny-detector_best.weights', 'sideview-yolov4-tiny-detector.cfg', 'sideview.names', 1, 'left')
        self.left_camera.image_update.connect(self.image_update_slot2)
        self.left_camera.notification_update.connect(self.notification_banner_update)
        self.right_camera = CameraFeed('sideview-yolov4-tiny-detector_best.weights', 'sideview-yolov4-tiny-detector.cfg', 'sideview.names', 2, 'right')
        self.right_camera.image_update.connect(self.image_update_slot3)
        self.right_camera.notification_update.connect(self.notification_banner_update)

        """ Connect Power Button """
        self.power = False
        self.startup()
        self.on_pushButton.setFlat(True)
        self.on_pushButton.clicked.connect(self.turn_on)

    """ Connect CameraFeed Signal to Label """
    def image_update_slot1(self, image):
        self.camera1.setPixmap(QPixmap.fromImage(image))

    def image_update_slot2(self, image):
        self.camera2.setPixmap(QPixmap.fromImage(image))

    def image_update_slot3(self, image):
        self.camera3.setPixmap(QPixmap.fromImage(image))
        
    """ Update Notification Banner """
    def notification_banner_update(self):
        # Global variables for CSV file
        global csv_file, csv_writer
        # Global variables for can classification
        global set_emit1, set_emit2, set_emit3

        # Compare and update banner
        acceptance = ''
        if set_emit1 and set_emit2 and set_emit3:
            self.notification_label.setStyleSheet(open('accepted.css').read())
            self.notification_label.setText('ACCEPTED')
            acceptance = 'GOOD'
        else:
            self.notification_label.setStyleSheet(open('rejected.css').read())
            self.notification_label.setText('REJECTED')
            acceptance = 'DAMAGED'
        
        # Write rejection/acceptance to CSV file
        current_time = datetime.datetime.now()
        csv_writer.writerow([acceptance, current_time])

    def startup(self):
        self.on_pushButton.setStyleSheet('QPushButton { background-image: url(off.png); }')
        self.logo_label.setStyleSheet('QLabel { image: url(NJFCO_logo.png);}')

    """ On/Off button click """
    def turn_on(self):
        if self.power:
            self.on_pushButton.setStyleSheet('QPushButton { background-image: url(off.png); }')
            self.notification_label.setStyleSheet(open('rejected.css').read())
            self.notification_label.setText('OFF')
            self.top_camera.stop()
            self.left_camera.stop()
            self.right_camera.stop()
            self.power = False
        else:
            self.on_pushButton.setStyleSheet('QPushButton { background-image: url(on.png); }')
            self.notification_label.setStyleSheet(open('accepted.css').read())
            self.notification_label.setText('ON')
            self.top_camera.start()
            self.left_camera.start()
            self.right_camera.start()
            self.power = True

""" Camera Feed Thread """
class CameraFeed(QThread):
    image_update = pyqtSignal(QImage)
    notification_update = pyqtSignal()

    def __init__(self, weights, config, labels, cam_index, pos) -> None:
        super().__init__()
        self.weights = weights
        self.config = config
        self.labels = labels
        self.capture = cv2.VideoCapture(cam_index)
        self.cam_position = pos
    
    def run(self):
        self.ThreadActive = True
        color = (255,255,255)   

        while self.ThreadActive:
            ret, frame = self.capture.read()
            #Lower framerate
            #count += 1
            #if count % 10 != 0:
            #    continue

            """ Call Object Detection on Image """
            if ret:
                # Resize image and detect object
                img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (400, 250))
                yolo = YOLO(self.weights, self.config, self.labels)
                bbox, label, conf = yolo.detect_objects(img)
                cam = yolo.draw_bbox(img, bbox, label, conf, color)

                """ Classify Detected Object and Update Notification"""
                try:
                    # In list of detected objects identify the most confident
                    can = conf.index(max(conf))
                    # Make sure confidence level is high enough
                    if conf[can] < .85:
                        continue
                    # Global variable for can classification
                    global set_emit1, set_emit2, set_emit3

                    # Set can global variable
                    if label[can] == 'Good can' and self.cam_position == 'top':
                        set_emit1 = True
                    elif label[can] == 'Damaged Can' and self.cam_position == 'top':
                        set_emit1 = False
                        print(label[can])
                    elif label[can] == 'good' and self.cam_position == 'left':
                        set_emit2 = True
                    elif label[can] == 'bad' and self.cam_position == 'left':
                        set_emit2 = False
                    elif label[can] == 'good' and self.cam_position == 'right':
                        set_emit3 = True
                    elif label[can] == 'bad' and self.cam_position == 'right':
                        set_emit3 = False
                    
                    # Update Banner
                    self.notification_update.emit()

                except ValueError:
                    print('No object detected')

                """ Recreate QT compatible image"""
                convert_to_QtFormat = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                pic = convert_to_QtFormat.scaled(400, 250, Qt.KeepAspectRatio)
                self.image_update.emit(pic)
                

    def stop(self):
        self.ThreadActive = False
        self.quit()
    

    
def main():
    app = QApplication([])
    Root = MainWindow()
    app.exec()

if __name__ == '__main__':
    set_emit1 = False
    set_emit2 = False
    set_emit3 = False
    csv_file = open('object_detection.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Object', 'Time' ])
    main()