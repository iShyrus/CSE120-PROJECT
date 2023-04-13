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

        """ Create Camera Feeds and Connect Labels"""
        self.top_camera = TopCameraFeedTinyYOLOV4()
        self.top_camera.image_update.connect(self.image_update_slot1)
        self.left_camera = CameraFeed()
        self.left_camera.image_update.connect(self.image_update_slot2)
        self.left_camera.notification_update.connect(self.notification_banner_update)
        #self.right_camera = CameraFeed()
        #self.right_camera.image_update.connect(self.image_update_slot)

        """ Connect Power Button """
        self.power = False
        self.startup()
        self.on_pushButton.setFlat(True)
        self.on_pushButton.clicked.connect(self.turn_on)

    """ Connect Camera to Label """
    # Connect Camera to Labelca
        # Connect Camera to Labelca
    def image_update_slot1(self, image):
        self.camera1.setPixmap(QPixmap.fromImage(image))

    def image_update_slot2(self, image):
        self.camera2.setPixmap(QPixmap.fromImage(image))

    #def image_update_slot3(self, image):
    #    self.camera3.setPixmap(QPixmap.fromImage(image))
        
    """ Update Notification Banner """
    def notification_banner_update(self, acceptance):
        if acceptance:
            self.notification_label.setStyleSheet(open('accepted.css').read())
            self.notification_label.setText("ACCEPTED")
        else:
            self.notification_label.setStyleSheet(open('rejected.css').read())
            self.notification_label.setText("REJECTED")

    def startup(self):
        self.on_pushButton.setStyleSheet("QPushButton { background-image: url(off.png); }")
        self.logo_label.setStyleSheet("QLabel { image: url(NJFCO_logo.png);}")

    """ On/Off button click """
    def turn_on(self):
        if self.power:
            self.on_pushButton.setStyleSheet("QPushButton { background-image: url(off.png); }")
            self.notification_label.setStyleSheet(open('rejected.css').read())
            self.notification_label.setText("OFF")
            #self.top_camera.stop()
            self.left_camera.stop()
            #self.right_camera.stop()
            self.power = False
        else:
            self.on_pushButton.setStyleSheet("QPushButton { background-image: url(on.png); }")
            self.notification_label.setStyleSheet(open('accepted.css').read())
            self.notification_label.setText("ON")
            #self.top_camera.start()
            self.left_camera.start()
            #self.right_camera.start()
            self.power = True

class TopCameraFeedTinyYOLOV4(QThread):
    image_update = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True

        cap = cv2.VideoCapture(0)
        weights = "top-detection_best.weights"
        config = "top-detection.cfg"
        labels = "top-detection.names"
        color = (255,255,255)
        count = 0

        csv_file = open('object_detection.csv', 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Object', 'Confidence', 'Time' ])


        while self.ThreadActive:
            ret,img=cap.read()
            #Lower framerate
            #count += 1
            #if count % 10 != 0:
            #    continue
            img=cv2.resize(img,(400,250))
            
            #Trained Model
            yolo = YOLO(weights, config, labels)
            
            #outputs boundary, label, and confidence of can when detected
            bbox, label, conf = yolo.detect_objects(img)
            
            for i in range(len(label)):
                if label[i] == 'Damaged Can' or label[i] == 'Good can':
                    if conf[i] > .98:
                        current_time = datetime.datetime.now()
                        csv_writer.writerow([label[i], conf[i], current_time])
            
            #draws boundary box at the location of can
            cam = yolo.draw_bbox(img, bbox, label, conf, color)
            
            try:
                # In list of detected objects identify the most confident
                can = conf.index(max(conf))

                # Emit to notification banner
                if label[can] == 'Good can':
                    self.notification_update.emit(True)
                else:
                    self.notification_update.emit(False)
            except ValueError:
                print("No object detected")
            
            convert_to_QtFormat = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            pic = convert_to_QtFormat.scaled(400, 250, Qt.KeepAspectRatio)
            self.image_update.emit(pic)


    def stop(self):
        self.ThreadActive = False
        self.quit()

""" Camera Feeds Threads """
class CameraFeed(QThread):
    image_update = pyqtSignal(QImage)
    notification_update = pyqtSignal(bool)
    
    def run(self):
        self.ThreadActive = True
        weights = "sideview-yolov4-tiny-detector_best.weights"
        config = "sideview-yolov4-tiny-detector.cfg"
        labels = "obj.names"
        color = (100,100,100)   
        capture = cv2.VideoCapture(1)

        while self.ThreadActive:
            ret, frame = capture.read()
            
            if ret:
                img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 320))
                yolo = YOLO(weights, config, labels)
                bbox, label, conf = yolo.detect_objects(img)
                cam = yolo.draw_bbox(img, bbox, label, conf, color)
                print(label)
                print(conf)

                try:
                    # In list of detected objects identify the most confident
                    can = conf.index(max(conf))

                    # Emit to notification banner
                    if label[can] == 'good':
                        self.notification_update.emit(True)
                    else:
                        self.notification_update.emit(False)
                except ValueError:
                    print("No object detected")

                """ Recreate QT compatible image"""
                flipped_img = cv2.flip(img, 1)
                convert_to_QtFormat = QImage(flipped_img.data, flipped_img.shape[1], flipped_img.shape[0], QImage.Format_RGB888)
                pic = convert_to_QtFormat.scaled(320, 320, Qt.KeepAspectRatio)
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
