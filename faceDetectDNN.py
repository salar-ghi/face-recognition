from cvzone.FaceDetectionModule import FaceDetector
import cv2
import torch
import os 
from numba import jit, cuda

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device('cuda')

# torch.zeros(1).cuda()

# @jit(parallel=True)
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

# 226
# 243
detector = FaceDetector()

rtspurl =  "rtsp://admin:ndcndc@192.168.10.226:554/channel1"
mohandesurl =  "rtsp://admin:admin4763@192.168.10.240:554/channel1"
Localurl =  'rtsp://admin:admin4763@192.168.5.190:554/'
httpurl =  'http://192.168.10.226:80/video'

resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]
res_index = 5
cam = cv2.VideoCapture(rtspurl)
cam.set(cv2.CAP_PROP_FPS, 25)
cam.set(cv2.CAP_PROP_FRAME_COUNT,1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[3][0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[3][1])


# print('fps:',cam.get(cv2.CAP_PROP_FPS))
# print('WIDTH:',cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# print('HEIGHT:',cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cam.read()
    # print(ret)
    frame, bboxs = detector.findFaces(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 3)
    #     roi_gray = gray[y: y + h , x: x + w]
    #     roi_color = frame[y: y + h , x: x + w]

    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        x2 = x1 + w1
        y2 = y1 + h1
    
    frm = ResizeWithAspectRatio(frame, width=1280, height=704)
    cv2.imshow('Real-time Detection', frm)
    k = cv2.waitKey(1) & 0Xff
    if k == 27: # Press 'ESC' to quit
        break
        
cam.release()
cv2.destroyAllWindows()