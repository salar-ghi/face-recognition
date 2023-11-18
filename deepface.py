from deepface import DeepFace

import os 
from cvzone.FaceDetectionModule import FaceDetector
import cv2
# import numpy as np


# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    ret, frame = cam.read()
    frame, bboxs = detector.findFaces(frame)
    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        x2 = x1 + w1
        y2 = y1 + h1
        
        verification = DeepFace.verify(frame)
    
    cv2.imshow('Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()