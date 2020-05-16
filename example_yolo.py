from yoloface import yoloFace
import cv2

"""
Get the Weight File 'model-weights/yolov3-wider_16000.weights' : 
    Download from the below link :
        https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view?usp=sharing
"""

cap = cv2.VideoCapture(0)
detector =  yoloFace()

while True:
    _, frame = cap.read()
    
    boxes = detector.detectFaces(frame)
    
    print("No. of faces Detected : {}".format(len(boxes)))
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
    
    cv2.imshow("Face", frame)
    key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    if key == 32: break

cap.release()