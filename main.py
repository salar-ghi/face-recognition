import os 
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import face_recognition
import numpy as np
import pickle
import cvzone

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, EmployeeIds = encodeListKnownWithIds

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



detector = FaceDetector()
# 226
# 243
rtspurl =  'rtsp://admin:ndcndc@192.168.10.226:554/channel1'
httpurl =  'http://192.168.10.226:80/video'
cam = cv2.VideoCapture(rtspurl)
# print(cam.grab())
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cam.read()
    if ret is False:
        break
    
    global x, y, w, h , x2
    # global matches, faceDis, matchIndex

    frame, bboxs = detector.findFaces(frame)
    for bbox in bboxs:
        x, y, w, h = bbox['bbox']
        x2 = x + (int(w) / 2)

    ########################## start to recognize ##########################
    imgSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, faceCurFrame)
    
    flag = False
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)        
        if faceDis[matchIndex] < 0.5 and matches[matchIndex]:
            # for bbox in bboxs:
            #     x, y, w, h = bbox['bbox']
            #     x2 = x + (int(w) / 2)
            cvzone.putTextRect(frame, f'{EmployeeIds[matchIndex]}', (int(x2+45), y-20),1, 1, (0, 255, 25),(10, 10, 10, 0.1), cv2.BORDER_TRANSPARENT,1, 1)
            cvzone.cornerRect(frame, (x, y, w, h))
        elif faceDis[matchIndex] > 0.5 or (not matches[matchIndex]):
            txt ="Unknown"
            # for bbox in bboxs:
            #     x, y, w, h = bbox['bbox']
            #     x2 = x + (int(w) / 2)
            cvzone.putTextRect(frame, f'{txt}', (int(x2+15), y-15),1, 1, (0, 0, 255),(255, 255, 255, 0.1), cv2.BORDER_TRANSPARENT,1, 1)
            cvzone.cornerRect(frame, (x, y, w, h))
            flag = True
            break 
        else:
            txt ="Unknown"
            # for bbox in bboxs:
            #     x, y, w, h = bbox['bbox']
            #     x2 = x + (int(w) / 2)
            cvzone.putTextRect(frame, f'{txt}', (int(x2+15), y-15),1, 1, (0, 0, 255),(255, 255, 255, 0.9), cv2.BORDER_TRANSPARENT,1, 1)
            cvzone.cornerRect(frame, (x, y, w, h))
            flag = True
            break 

            
    # if flag:
    #     txt ="Unknown"
    #     for bbox in bboxs:
    #         x, y, w, h = bbox['bbox']
    #         x2 = x + (int(w) / 2)
    #         cvzone.putTextRect(frame, f'{txt}', (int(x2+15), y-15),1, 1, (0, 255, 25),(10, 10, 10, 0.1), cv2.BORDER_TRANSPARENT,1, 1)
    #         cvzone.cornerRect(frame, (x, y, w, h))
    frm = ResizeWithAspectRatio(frame, width=960)
    cv2.imshow("Real-time Detection", frm)

    ########################## recognition completed ##########################
    k = cv2.waitKey(30) & 0Xff
    if k == 27: # Press 'ESC' to quit
        cam.release()
        cv2.destroyAllWindows()
        break