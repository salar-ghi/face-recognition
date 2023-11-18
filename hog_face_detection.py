from helpers import convert_and_trim_bb
import argparse

import time
# from cvzone.FaceDetectionModule import FaceDetector
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--upsample", type=int, default=1, help="# of times to upsample")
args = vars(ap.parse_args())

cam = cv2.VideoCapture(0)

# load dlib's HOG + Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")

while True:
    ret, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    
    
    image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # perform face detection using dlib's face detector
    start = time.time()
    print("[INFO[ performing face detection with dlib...")
    faces = detector(gray, 1)

    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))
    boxes = [convert_and_trim_bb(image, r) for r in faces]
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()