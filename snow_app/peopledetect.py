# HOG 보행자 검출

# HOG(Histogram of Oriented Gradients)
# 영상의 지역적 그래디언트 방향 정보를 특징 벡터로 사용 
# 다양한 객체 인식에서 활용

# 순서
# 1. 입력영상에서 부분영상 추출
# 2. 크기 정규화(64x128)
# 3. 그래디언트 계산(방향&크기 성분 계산)
# 4. 그래디언트 영상을 8x8 크기의 셀 분할
# 5. 각 셀마다 방향과 크기 성분을 이용하여 방향 히스토그램 계산


import sys
import random
import numpy as np
import cv2


# 동영상 불러오기
cap = cv2.VideoCapture('.\snow_app.\\vtest.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 보행자 검출을 위한 HOG 기술자 설정
hog = cv2.HOGDescriptor() # HOG 기술자 객체 생성 및 보행자 검출을 위해 학습된 분류기 계수 불러오기
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) #SVM 분류기 계수 등록하기

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 매 프레임마다 보행자 검출
    detected, _ = hog.detectMultiScale(frame) # HOG 멀티스케일 객체 검출 함수(사각형 정보 받아온다.)

    # 검출 결과 화면 표시
    for (x, y, w, h) in detected:
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(frame, (x, y, w, h), c, 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
