import os 
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import face_recognition
import numpy as np
import pickle
import cvzone

print("Loading Encoded file .......")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()

encodeListKnown, EmployeeIds = encodeListKnownWithIds

# url = "http://admin:ndcndc@192.168.10.243/1"
cam = cv2.VideoCapture("rtsp://admin:ndcndc@192.168.10.243/")
cam.set(3, 1020) # set width
cam.set(4, 840) # set height

detector = FaceDetector()

while True:
    _, frame = cam.read()
    print('cannot read data from camera')
    if _ is False:
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ########################## start to recognize ##########################
    imgSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgSmall)
    encodeCurFrame = face_recognition.face_encodings(imgSmall, faceCurFrame)


    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        frame, bboxs = detector.findFaces(frame)

        if bboxs:
            for bbox in bboxs:
                # center = bbox["center"]
                x1, y1, w1, h1 = bbox['bbox']
                x2 = x1 + w1
                y2 = y1 + h1
                if faceDis[matchIndex] < 0.45 and matches[matchIndex]:  
                    # print("Known User :", name)
                    cv2.circle(frame, (x1, y1), 5, (255, 0, 75), cv2.BORDER_WRAP,)
                    cvzone.putTextRect(frame, f'{EmployeeIds[matchIndex]}', (x1, y1),2, 2, (0, 255, 50))
                    frame = cvzone.cornerRect(frame, (x1, y1, w1, h1))
                else:
                    txt = "Unknown"
                    cv2.circle(frame, (x1, y1), 5, (0 , 25, 255), cv2.BORDER_WRAP)
                    cvzone.putTextRect(frame, f'{txt}', (x1, y1), 2, 2, (0, 255, 45))
                    frame = cvzone.cornerRect(frame, (x1, y1, w1, h1))

                cv2.imshow("Real-time Detection", frame)

    ########################## recognition completed ##########################
    k = cv2.waitKey(30) & 0Xff
    if k == 27: # Press 'ESC' to quit
        break
        
cam.release()
cv2.destroyAllWindows()