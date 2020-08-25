# 캐스케이드 분류기: 얼굴 검출
# Viola - Jones 얼굴 검출기
# Positive 영상(얼굴 영상)과 negative 영상(얼굴 아닌 영상)을 훈련하여 빠르고 정확하게 얼굴 영역을 검출

# 유사하르 특징(Harr-like features)
# 흰색 사각형 영역 픽셀 값의 합에서 검정색 사각형 영역 픽셀 값을 뺀 결과 값을 추출

# 캐스케이드 분류기(Cascade classifier)
# Non-face 영역을 빠르게 skip하도록 다단계 검사 수행

import sys
import numpy as np
import cv2


src = cv2.imread('.\snow_app.\lenna.bmp')

if src is None:
    print('Image load failed!')
    sys.exit()

#객체 생성 및 학습 데이터를 불러와서 찾고자 하는 객체를 검출하는 기능 (정면 얼굴 검출)
classifier = cv2.CascadeClassifier('.\snow_app.\haarcascade_frontalface_alt2.xml')

if classifier.empty():
    print('XML load failed!')
    sys.exit()


tm = cv2.TickMeter()#시간 재기
tm.start()#시간 재기

# 멀티스케일 객체 검출 함수
faces = classifier.detectMultiScale(src, scaleFactor = 1.2, minSize = (100,100))

tm.stop()#시간 재기
print(tm.getTimeMilli()) #시간 재기

for (x, y, w, h) in faces:
    cv2.rectangle(src, (x, y, w, h), (255, 0, 255), 2)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
