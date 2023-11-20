from helpers import convert_and_trim_bb
import time
import dlib
import cv2
import imutils

# ap = argparse.ArgumentParser()
# ap.add_argument("-u", "--upsample", type=int, default=1, help="# of times to upsample")
# args = vars(ap.parse_args())

cam = cv2.VideoCapture('rtsp://admin:ndcndc@192.168.10.226:554/channel1')
# cam = cv2.VideoCapture('http://admin:ndcndc@192.168.10.243:554/channel1')

while True:
    ret, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    
    
    image = imutils.resize(image, width=1000)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(gray, 1)
    boxes = [convert_and_trim_bb(image, r) for r in faces]
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()