import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5.QtCore import QTimer

import cv2
from cvlib.object_detection import YOLO
import csv
import datetime
from ultralytics import YOLO as YOLOV8
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import time

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("can_detection_gui.ui", self)
        self.show()

        """ Connect Power Button """
        self.power = False
        self.startup()
        self.on_pushButton.setFlat(True)
        self.on_pushButton.clicked.connect(self.turn_on)

        """ Button Switch Tiny-YOLO / YOLOv8"""
        self.tiny_button.clicked.connect(self.tinyYoloCheck)
        self.yolov8_button.clicked.connect(self.yolov8Check)

        """Timer for FPS Update"""
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.fps_update)
        self.timer.start()

        """ Create CameraFeed Objects, Default to TinyYolo (less demanding)"""
        self.top_camera = CameraFeed('top-detection_best.weights', 'top-detection.cfg', 'top-detection.names', 2, 'top',"tinyYolo")
        self.top_camera.image_update.connect(self.image_update_slot1)
        self.top_camera.notification_update.connect(self.notification_banner_update)
        self.left_camera = CameraFeed('sideview-yolov4-tiny-detector_best.weights', 'sideview-yolov4-tiny-detector.cfg', 'sideview.names', 1, 'left',"tinyYolo")
        self.left_camera.image_update.connect(self.image_update_slot2)
        self.left_camera.notification_update.connect(self.notification_banner_update)
        self.right_camera = CameraFeed('sideview-yolov4-tiny-detector_best.weights', 'sideview-yolov4-tiny-detector.cfg', 'sideview.names', 0, 'right',"tinyYolo")
        self.right_camera.image_update.connect(self.image_update_slot3)
        self.right_camera.notification_update.connect(self.notification_banner_update)

    """Tiny Yolo"""
    def tinyYoloCheck(self):
        self.top_camera = CameraFeed('top-detection_best.weights', 'top-detection.cfg', 'top-detection.names', 2, 'top',"tinyYolo")
        self.top_camera.image_update.connect(self.image_update_slot1)
        self.top_camera.notification_update.connect(self.notification_banner_update)
        self.left_camera = CameraFeed('sideview-yolov4-tiny-detector_best.weights', 'sideview-yolov4-tiny-detector.cfg', 'sideview.names', 1, 'left',"tinyYolo")
        self.left_camera.image_update.connect(self.image_update_slot2)
        self.left_camera.notification_update.connect(self.notification_banner_update)
        self.right_camera = CameraFeed('sideview-yolov4-tiny-detector_best.weights', 'sideview-yolov4-tiny-detector.cfg', 'sideview.names', 0, 'right',"tinyYolo")
        self.right_camera.image_update.connect(self.image_update_slot3)
        self.right_camera.notification_update.connect(self.notification_banner_update)

    """YOLOv8"""
    def yolov8Check(self):
        self.top_camera = CameraFeed(YOLOV8('topYoloV8Weights.pt'), '', '', 2, 'top',"yolov8")
        self.top_camera.image_update.connect(self.image_update_slot1)
        self.top_camera.notification_update.connect(self.notification_banner_update)
        self.left_camera = CameraFeed(YOLOV8('yolov8side.pt'), '', '', 1, 'left',"yolov8")
        self.left_camera.image_update.connect(self.image_update_slot2)
        self.left_camera.notification_update.connect(self.notification_banner_update)
        self.right_camera = CameraFeed(YOLOV8('yolov8side.pt'), '', '', 0, 'right',"yolov8")
        self.right_camera.image_update.connect(self.image_update_slot3)
        self.right_camera.notification_update.connect(self.notification_banner_update)


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
    
    """FPS Update"""
    def fps_update(self):
        global set_fps
        self.fps_label.setText("FPS: "+str(set_fps))
        print(set_fps)

""" Camera Feed Thread """
class CameraFeed(QThread):
    image_update = pyqtSignal(QImage)
    notification_update = pyqtSignal()

    def __init__(self, weights, config, labels, cam_index, pos ,model) -> None:
        super().__init__()
        self.weights = weights
        self.config = config
        self.labels = labels
        self.capture = cv2.VideoCapture(cam_index)
        self.cam_position = pos
        self.model = model

    def run(self):
        self.ThreadActive = True
        color = (255,255,255)
        fps_start_time = 1
        fps = 1

        while self.ThreadActive:
            ret, frame = self.capture.read()
            #Lower framerate
            #count += 1
            #if count % 10 != 0:
            #    continue

            """ Call Object Detection on Image """
            if ret:
                img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (400, 250))

                """FPS Check"""
                fps_end_time = time.time()
                time_diff = fps_end_time-fps_start_time
                fps=1/(time_diff)
                fps_start_time = fps_end_time
                global set_fps  
                set_fps = fps

                # Resize image and detect object
                if self.model =="tinyYolo":
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

                if self.model == "yolov8":

                    model = self.weights
                    detect_params = model.predict(source=[frame], conf=0.9, device = "0")
                    checkCan ="Nothing"
                    for r in detect_params:
                        confidence = str(r.boxes.conf)
                        confidence = confidence.replace("tensor([","")
                        confidence = confidence.replace("], device='cuda:0')","")
                        for c in r.boxes.cls:
                            checkCan = model.names[int(c)]


                    """ Classify Detected Object and Update Notification"""
                    try:
                        # Make sure confidence level is high enough
                        if int(confidence) < .85:
                            continue
                        # Global variable for can classification

                        # Set can global variable
                        if checkCan == 'Good can' and self.cam_position == 'top':
                            set_emit1 = True
                        elif checkCan == 'Damaged Can' and self.cam_position == 'top':
                            set_emit1 = False
                        elif checkCan == 'good' and self.cam_position == 'left':
                            set_emit2 = True
                        elif checkCan == 'bad' and self.cam_position == 'left':
                            set_emit2 = False
                        elif checkCan == 'good' and self.cam_position == 'right':
                            set_emit3 = True
                        elif checkCan == 'bad' and self.cam_position == 'right':
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
    set_fps = 0
    csv_file = open('object_detection.csv', 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Object', 'Time' ])
    main()
