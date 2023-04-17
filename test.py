from cvlib.object_detection import YOLO
import cv2
import csv
import datetime

cap = cv2.VideoCapture(0)
weights = "top-detection_best.weights"
config = "top-detection.cfg"
labels = "top-detection.names"
color = (255,255,255)
count = 0

csv_file = open('object_detection.csv', 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Object', 'Confidence', 'Time' ])

while True:
    ret,img=cap.read()
    #Lower framerate
    #count += 1
    #if count % 10 != 0:
    #    continue
    img=cv2.resize(img,(680,460))
    
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
    
    
    cv2.imshow("cam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
csv_file.close()
cv2.destroyAllWindows()