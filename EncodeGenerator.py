import pickle
import os
import sys
import face_recognition
import cv2

folderPath = 'dataset'
pathList = os.listdir(folderPath)
# print(pathList)

imgList = []
EmployeeIds = []


for item in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, item)))
    EmployeeIds.append(os.path.splitext(item)[0])

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        # imgS = cv2.resize(img, (0, 0), None, 1, 1)
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # face = face_recognition.face_locations(imgS)
        encode = face_recognition.face_encodings(imgS)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started ... ")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, EmployeeIds]
print("Encoding Complete")

file = open("EncodeFile.py", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()