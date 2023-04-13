import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import cv2
import os
import time

import numpy as np
import csv
from datetime import date
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("can_detection_gui.ui", self)
        self.show()

        # Create Camera Feeds
        self.top_camera = TopCameraFeedYOLOV8()
        self.top_camera.image_update.connect(self.image_update_slot1)
        self.left_camera = sideCameraFeedYOLOV8()
        self.left_camera.image_update.connect(self.image_update_slot2)
        # self.right_camera = CameraFeed()
        # self.right_camera.image_update.connect(self.image_update_slot)

        # Connect Power Button
        self.power = False
        self.startup()
        self.on_pushButton.setFlat(True)
        self.on_pushButton.clicked.connect(self.turn_on)

    # Connect Camera to Labelca
    def image_update_slot1(self, image):
        self.camera1.setPixmap(QPixmap.fromImage(image))
    def image_update_slot2(self, image):
        self.camera2.setPixmap(QPixmap.fromImage(image))
        
        # self.camera3.setPixmap(QPixmap.fromImage(image))

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
            # self.right_camera.stop()
            self.power = False
        else:
            self.on_pushButton.setStyleSheet("QPushButton { background-image: url(on.png); }")
            self.notification_label.setStyleSheet(open('accepted.css').read())
            self.notification_label.setText("ACCEPTED")
            self.top_camera.start()
            self.left_camera.start()
            # self.right_camera.start()
            self.power = True

# Camera Feeds Threads

class TopCameraFeedYOLOV8(QThread):
    image_update = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True

        capture = cv2.VideoCapture(2)
        countCan = 0
        fps_start_time = 0
        fps = 0
        confidence = "0"
        checkCan = "Nothing"
        temp = ""
        csvCheck = False
        didCSV = False
        model = YOLO("topYoloV8Weights.pt")
        #For ARUCO detection
        parameters =  cv2.aruco.DetectorParameters()
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        while self.ThreadActive:
            ret, frame = capture.read()
            # Create QT compatible image if no issue

            if ret:
                #FPS Check
                fps_end_time = time.time()
                time_diff = fps_end_time-fps_start_time
                fps=1/(time_diff)
                fps_start_time = fps_end_time
                fps_text = "FPS: {:.2f}".format(fps)
                if confidence!="":
                    confidenceText = "Confidence: {}".format((confidence))
                else:
                    confidenceText = "Confidence: None"
                cv2.putText(frame,fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),1) 
                cv2.putText(frame,checkCan,(5,55),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),1) 
                cv2.putText(frame,confidenceText,(5,80),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),1) 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(5,5),0)
                circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=110, param1=40, param2=30, minRadius=160, maxRadius=170)  

                img_copy = frame.copy()

                #Detects good/bad can along with confidence level
                detect_params = model.predict(source=[frame], conf=0.9, device = "0")
                checkCan ="Nothing"
                for r in detect_params:
                    confidence = str(r.boxes.conf)   
                    confidence = confidence.replace("tensor([","")
                    confidence = confidence.replace("], device='cuda:0')","")

                    for c in r.boxes.cls:
                        print(model.names[int(c)])
                        # if model.names[int(c)] =="0":
                        #     checkCan = "Good Can"
                        # if model.names[int(c)] =="Damage Can":
                        #     checkCan = "Damaged Can"
                        checkCan = model.names[int(c)]



                #Detects ARUCO marker and makes lines
                corners, markerIds, rejectedCandidates = detector.detectMarkers(img_copy)
                int_corners = np.int0(corners)
                cv2.polylines(img_copy, int_corners, True, (0,255,0),2)
                if corners:
                    aruco_perimeter = cv2.arcLength(corners[0],True)

                #Pixel to cm perimeter ratio
                    pixelToCm = aruco_perimeter / 19.6

                #Detect can circles and diameter 
                # if circles is not None:
                #     print(didCSV)
                #     if didCSV==False:
                #         if checkCan!="Nothing":
                #             countCan+=1
                #             today = date.today()
                #             t = time.localtime()
                #             currentTime = time.strftime("%H:%M:%S",t)
                #             dict = {"Date":date.today(),"Time":currentTime,"Can Number":str(countCan),"Can Lid Status":checkCan,"Can Side Status":"Good"}
                #             fieldNames = ["Date","Time","Can Number","Can Lid Status","Can Side Status"]

                #             with open("cans.csv", "a", newline ="") as csv_file:
                #                 dict_object = csv.DictWriter(csv_file, fieldnames=fieldNames) 
                #                 dict_object.writerow(dict)
                        
                        
                #             didCSV = True

                #     circles = np.round(circles[0, :]).astype("int")
                #     for (x, y, r) in circles:
                #         if pixelToCm:   
                #             objectDiameter = (r/pixelToCm)
                #         cv2.circle(img_copy, (x, y), r, (0, 255, 0), 2)
                #         cv2.circle(img_copy, (int(x),int(y)),5,(0,0,255),-1)
                #         cv2.putText(img_copy, "Radius {}cm".format(round(objectDiameter,2)), (int(x),int(y-15)), cv2.FONT_HERSHEY_PLAIN,2, (100,200,0),2)

                elif circles is None:
                    didCSV=False
                    print("test")

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convert_to_QtFormat = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                pic = convert_to_QtFormat.scaled(400, 250, Qt.KeepAspectRatio)
                self.image_update.emit(pic)




    def stop(self):
        self.ThreadActive = False
        self.quit()

    
class sideCameraFeedYOLOV8(QThread):
    image_update = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True
        capture = cv2.VideoCapture(5)
        model = YOLO("yolov8side.pt")
        checkCan = "Nothing"

        while self.ThreadActive:
            ret, frame = capture.read()

            if ret:
                cv2.putText(frame,checkCan,(5,55),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),1) 

                #Detects good/bad can along with confidence level
                detect_params = model.predict(source=[frame], conf=0.9)
                checkCan ="Nothing"
                for r in detect_params:
                    for c in r.boxes.cls:
                        checkCan = model.names[int(c)]

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