import sys
import numpy as np
import cv2


src = cv2.imread('.\snow_app.\lenna.bmp')

if src is None:
    print('Image load failed!')
    sys.exit()

face_classifier = cv2.CascadeClassifier('.\snow_app.\haarcascade_frontalface_alt2.xml')#얼굴 검출
eye_classifier = cv2.CascadeClassifier('.\snow_app.\haarcascade_eye.xml')#눈 검출

if face_classifier.empty() or eye_classifier.empty():
    print('XML load failed!')
    sys.exit()

faces = face_classifier.detectMultiScale(src) #얼굴 검출

for (x1, y1, w1, h1) in faces:
    cv2.rectangle(src, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 255), 2)

    faceROI = src[y1:y1 + h1 // 2, x1:x1 + w1] # 얼굴 반쪽 위에서만
    eyes = eye_classifier.detectMultiScale(faceROI) #눈 검출(faceROI 범위에서)

    for (x2, y2, w2, h2) in eyes:
        center = (x2 + w2 // 2, y2 + h2 // 2)
        cv2.circle(faceROI, center, w2 // 2, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
