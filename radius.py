import cv2
import numpy as np
import csv
from datetime import date
import time
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

#YOLO Model
model = YOLO("best2.pt")


#For ARUCO detection
parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

#Image loader
# img = cv2.imread('can9aruco.jpg', cv2.IMREAD_UNCHANGED) 

#Video loader
cap = cv2.VideoCapture(0)


#Scale Image
# scale_percent = 30 
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

countCan = 0
fps_start_time = 0
fps = 0
confidence = "0"
checkCan = "Nothing"
temp = ""
csvCheck = False
didCSV = False
#Alter image for detection
while True:
    ret, frame = cap.read()

    #FPS Check
    fps_end_time = time.time()
    time_diff = fps_end_time-fps_start_time
    fps=1/(time_diff)
    fps_start_time = fps_end_time
    fps_text = "FPS: {:.2f}".format(fps)

    #Confidence Check
    if confidence!="":
        confidenceText = "Confidence: {}".format((confidence))
    else:
        confidenceText = "Confidence: None"


    #Text 
    cv2.putText(frame,fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),1) 
    cv2.putText(frame,checkCan,(5,55),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),1) 
    cv2.putText(frame,confidenceText,(5,80),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),1) 

    #Helps detect circles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=110, param1=40, param2=30, minRadius=160, maxRadius=170)

    img_copy = frame.copy()

    #Detects good/bad can along with confidence level
    detect_params = model.predict(source=[frame], conf=0.9)
    checkCan ="Nothing"
    for r in detect_params:
        confidence = str(r.boxes.conf)   
        confidence = confidence.replace("tensor([","")
        confidence = confidence.replace("], device='cuda:0')","")

        for c in r.boxes.cls:
            if model.names[int(c)] =="0":
                checkCan = "Good Can"
            if model.names[int(c)] =="Damage Can":
                checkCan = "Damaged Can"



    #Detects ARUCO marker and makes lines
    corners, markerIds, rejectedCandidates = detector.detectMarkers(img_copy)
    int_corners = np.int0(corners)
    cv2.polylines(img_copy, int_corners, True, (0,255,0),2)
    if corners:
        aruco_perimeter = cv2.arcLength(corners[0],True)

    #Pixel to cm perimeter ratio
        pixelToCm = aruco_perimeter / 19.6


    #Detect can circles and diameter 
    if circles is not None:
        print(didCSV)
        if didCSV==False:
            if checkCan!="Nothing":
                countCan+=1
                today = date.today()
                t = time.localtime()
                currentTime = time.strftime("%H:%M:%S",t)
                dict = {"Date":date.today(),"Time":currentTime,"Can Number":str(countCan),"Can Lid Status":checkCan,"Can Side Status":"Good"}
                fieldNames = ["Date","Time","Can Number","Can Lid Status","Can Side Status"]

                with open("cans.csv", "a", newline ="") as csv_file:
                    dict_object = csv.DictWriter(csv_file, fieldnames=fieldNames) 
                    dict_object.writerow(dict)
            
            
                didCSV = True

        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if pixelToCm:   
                objectDiameter = (r/pixelToCm)
            cv2.circle(img_copy, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img_copy, (int(x),int(y)),5,(0,0,255),-1)
            cv2.putText(img_copy, "Radius {}cm".format(round(objectDiameter,2)), (int(x),int(y-15)), cv2.FONT_HERSHEY_PLAIN,2, (100,200,0),2)

    elif circles is None:
        didCSV=False
        print("test")
        







    
    


    cv2.imshow("Resized image", img_copy)

    # q to quit
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
