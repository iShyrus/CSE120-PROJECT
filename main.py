import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic
import cv2
import os
import time
import re
from tflite_runtime.interpreter import Interpreter
import numpy as np

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("can_detection_gui.ui", self)
        self.show()

        """ Create Camera Feeds and Connect Labels"""
        #self.top_camera = CameraFeed()
        #self.top_camera.image_update.connect(self.image_update_slot)
        self.left_camera = CameraFeed()
        self.left_camera.image_update.connect(self.image_update_slot)
        self.left_camera.notification_update.connect(self.notification_banner_update)
        #self.right_camera = CameraFeed()
        #self.right_camera.image_update.connect(self.image_update_slot)

        """ Connect Power Button """
        self.power = False
        self.startup()
        self.on_pushButton.setFlat(True)
        self.on_pushButton.clicked.connect(self.turn_on)

    """ Connect Camera to Label """
    def image_update_slot(self, image):
        #self.camera1.setPixmap(QPixmap.fromImage(image))
        self.camera2.setPixmap(QPixmap.fromImage(image))
        #self.camera3.setPixmap(QPixmap.fromImage(image))
        
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
        


""" Tensor Functions and Object Detection """
def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)
    
def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor
    
def detect_objects(interpreter, image, threshold):
    """ Returns a list of detection results, each a dictionary of object info """
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))
    
    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


""" Camera Feeds Threads """
class CameraFeed(QThread):
    image_update = pyqtSignal(QImage)
    notification_update = pyqtSignal(bool)
    
    def run(self):
        self.ThreadActive = True
        
        interpreter = Interpreter('detect.tflite')
        interpreter.allocate_tensors()
        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

        
        capture = cv2.VideoCapture(0)

        while self.ThreadActive:
            ret, frame = capture.read()
            """ Create QT compatible image if no issue """
            if ret:
                img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 320))
                res = detect_objects(interpreter, img, 0.2)
                print(res)
                
                for result in res:
                    if result['score'] > 0.8:
                        ymin, xmin, ymax, xmax = result['bounding_box']
                        xmin = int(max(1,xmin * CAMERA_WIDTH))
                        xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
                        ymin = int(max(1, ymin * CAMERA_HEIGHT))
                        ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
                        
                        img = cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
                        img = cv2.putText(frame, "GoodCan", (xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 
                        self.notification_update.emit(True)
                    else:
                        self.notification_update.emit(False)
                
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
