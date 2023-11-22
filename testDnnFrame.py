from cvzone.FaceDetectionModule import FaceDetector
import cv2
import torch
from numba import jit, cuda

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device('cuda')

import tensorflow as tf
tf.test.gpu_device_name()

# @jit(nopython=False, parallel=True,fastmath=True, target_backend="cuda")
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

rtspurl =  "rtsp://admin:ndcndc@192.168.10.226:554/channel1"
Localurl =  'rtsp://admin:admin4763@192.168.5.190:554/'
httpurl =  'http://192.168.10.226:80/video'

resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]
cam = cv2.VideoCapture(Localurl)
cam.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FRAME_COUNT,1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[0][0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[0][1])

while True:
    ret, frame = cam.read()
    print(frame)
    if ret is False:
        print('app enounter a error')
        break
    frame, bboxs = detector.findFaces(frame)
    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']        
    frm = ResizeWithAspectRatio(frame, width=1280, height=704)
    cv2.imshow('Real-time Detection', frm)
    k = cv2.waitKey(1) & 0Xff
    if k == 27: # Press 'ESC' to quit
        cam.release()
        break

cv2.destroyAllWindows()