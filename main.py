import os 
import cv2
# import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set width
cam.set(4, 480) # set height 


while True:
    ret, frame = cam.read()
    # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces  = face_cascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # for (x, y, w, h) in faces:
    #     face_roi = grayFrame[y:y + h, x:x + w]
    #     resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    #     normalized_face = resized_face / 255.0
    #     reshaped_face = normalized_face.reshape(1, 48, 48, 1)
    #     # preds = model.predict(reshaped_face)[0]
    #     # emotion_idx = preds.argmax()
    #     # emotion = emotion_labels[emotion_idx]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     # cv2.putText(frame, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    cv2.imshow('Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()