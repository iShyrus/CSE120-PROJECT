import cv2
import numpy as np
import csv
from datetime import date
import time
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

#YOLO Model
model = YOLO("topYoloV8Weights.pt")



cap = cv2.VideoCapture(2)

checkCan = "Nothing"


while True:
    ret, frame = cap.read()
    cv2.putText(frame,checkCan,(5,55),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),1) 


    #Detects good/bad can along with confidence level
    detect_params = model.predict(source=[frame], conf=0.9)
    checkCan ="Nothing"
    for r in detect_params:
        for c in r.boxes.cls:
            checkCan = model.names[int(c)]


    cv2.imshow("Resized image", frame)

    # q to quit
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
