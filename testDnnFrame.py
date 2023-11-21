from cvzone.FaceDetectionModule import FaceDetector
import cv2
import torch
import multiprocessing as mp
import os 
from numba import jit, cuda
from threading import Thread
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device('cuda')

class DNNDetect():

    def __init__(self, connUrl):
        #load pipe for data transmittion to the process
        print('enter the class')
        self.parent_conn, child_conn = mp.Pipe()
        self.p = mp.Process(target=self.mainFrame, args=(child_conn, connUrl))

        print('call the main method 1')
        #start process
        self.p.daemon = True
        self.p.start()
        print('call the main method2')


        # mainFrame()

    # @jit(nopython=False, parallel=True,fastmath=True, target_backend="cuda")
    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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


    # @jit(nopython=False, parallel=True, target_backend="cuda")
    def capture(self, connUrl):
        print('capture method run')
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()

        resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]
        cam = cv2.VideoCapture(connUrl)
        cam.set(cv2.CAP_PROP_FPS, 25)
        cam.set(cv2.CAP_PROP_FRAME_COUNT,1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[3][0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[3][1])
        return cam


    # @jit(nopython=False, parallel=True, target_backend="cuda")
    def showWindow(self, frameSource):
        cv2.imshow('Real-time Detection', frameSource)


    # @jit(nopython=False, parallel=True, fastmath=True,target_backend="cuda")
    def mainFrame(self, conn, connUrl):
        print('enter main method')
        rec_dat = conn.recv()
        self.parent_conn.send(1)
        self.parent_conn.recv()

        detector = FaceDetector()
        cap = self.parent_conn.recv()
        # cap = self.capture(connUrl)
        while True:
            ret, frame = cap.read()
            if ret is False:
                print('app enounter a error')
                break
            
            frame, bboxs = detector.findFaces(frame)

            if bboxs:
                x1, y1, w1, h1 = bboxs[0]['bbox']
            
            frm = self.ResizeWithAspectRatio(frame, width=1280, height=704)
            self.showWindow(frm)
            k = cv2.waitKey(1) & 0Xff
            if k == 27: # Press 'ESC' to quit
                cap.release()
                break

        cv2.destroyAllWindows()


    def end(self):        
        self.parent_conn.send(2)


if __name__ == '__main__':
    # def __init__(self, ):
    #load pipe for data transmittion to the process
    # self.parent_conn, child_conn = mp.Pipe()
    rtspurl =  "rtsp://admin:ndcndc@192.168.10.226:554/channel1"
    Localurl =  'rtsp://admin:admin4763@192.168.5.190:554/'
    httpurl =  'http://192.168.10.226:80/video'
    print('app starting')
    DNNDetect(Localurl)