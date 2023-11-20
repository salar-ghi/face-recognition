import os 
# from cvzone.FaceDetectionModule import FaceDetector
import cv2
# import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 226
# 243


cam = cv2.VideoCapture('rtsp://admin:admin4763@192.168.5.190:554/')
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960) 
# detector = FaceDetector()

while True:
    ret, frame = cam.read()
    print(ret)
    # _, bboxs = detector.findFaces(frame)
    # print(bboxs[0]['bbox'])


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 3)
        roi_gray = gray[y: y + h , x: x + w]
        roi_color = frame[y: y + h , x: x + w]

    # if bboxs:
    #     x1, y1, w1, h1 = bboxs[0]['bbox']
    #     x2 = x1 + w1
    #     y2 = y1 + h1
    
    frm = ResizeWithAspectRatio(frame, width=1280)
    cv2.imshow('Real-time Detection', frm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()