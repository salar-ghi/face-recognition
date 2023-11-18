from deepface import DeepFace
import os 
from cvzone.FaceDetectionModule import FaceDetector
import cv2
# import numpy as np

folderPath = 'dataset'
pathList= os.listdir(folderPath)
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]


imgList = []
EmployeeIds = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    EmployeeIds.append(os.path.splitext(path)[0])

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
detector = FaceDetector()
frames = []
while True:
    ret, frame = cam.read()
    frame, bboxs = detector.findFaces(frame)
    faces = DeepFace.detectFace(frame)
    if len(faces) > 0:
        frames.append(frame)
    if bboxs:        
        for img in imgList:
            verification = DeepFace.verify(frame, cv2.imread(img[0]) , model_name = "Facenet")
            print("thats' right :", verification[0])

            # if verification[0] == True:
            #     print("thats' right ")
            #     x1, y1, w1, h1 = bboxs[0]['bbox']
            #     x2 = x1 + w1
            #     y2 = y1 + h1
    
    cv2.imshow('Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()