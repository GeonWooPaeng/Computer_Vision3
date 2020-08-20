# 외곽선 검출
# 객체의 외곽선 좌표를 모두 추출하는 작업 
# 바깥쪽 & 안쪽(홀) 외곽선 -> 외곽선의 계층 구조도 표현 가능

import sys
import random
import numpy as np
import cv2


src = cv2.imread('.\Business_card_recognition_program.\contours.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

#src: 이진영상
contours, hier = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # 외곽선 검출 함수 

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) #출력영상을 color로 보여주기 위함

idx = 0
while idx >= 0:
    #특정 외곽선을 그리기 위함
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst, contours, idx, c, 2, cv2.LINE_8, hier) # 외곽선을 화면에 그리기(검증), hier을 줘서 안쪽까지 나타낸다.
    idx = hier[0, idx, 0] #hierarchy로 외곽선 계층 정보

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
