from cvlib.object_detection import YOLO
import cv2

cap = cv2.VideoCapture(0)
weights = "custom-yolov4-tiny-detector_final.weights"
config = "custom-yolov4-tiny-detector.cfg"
labels = "obj.names"
color = (255,255,255)
count = 0
while True:
    ret,img=cap.read()
    #Lower framerate
    #count += 1
    #if count % 10 != 0:
    #    continue
    img=cv2.resize(img,(680,460))
    
    yolo = YOLO(weights, config, labels)
    bbox, label, conf = yolo.detect_objects(img)
    cam = yolo.draw_bbox(img, bbox, label, conf, color)
    cv2.imshow("cam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()