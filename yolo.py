from ultralytics import YOLO
# from imutils.video import VideoStream
import cv2
import math 




model = YOLO("yolo-Weights/yolov8n.pt")

# Load the YOLOv8 model
# model = cv2.dnn.readNetFromDarknet("yolo-Weights/yolov8n.pt")
# Get the names of the output layers
# layer_names = model.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

classNames = ["human face"]

cap = cv2.VideoCapture(0)
# vs = VideoStream(src=0).start()
cap.set(3, 640)
cap.set(4, 480)


while True:
    ret, img = cap.read()
    # Perform object detection on the frame
        
    results = model(img, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            # cls = int(box.cls[0])
            cls = int(1)
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
# vs.stop()
cv2.destroyAllWindows()