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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started ... ")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, EmployeeIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()