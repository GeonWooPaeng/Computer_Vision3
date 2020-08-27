# 코너 검출
# 코너
# - 영상안에서 그림이 나타날때 객체의 뾰족한 부분
# - 고유한 특징을 가지고 있다(구분하기 쉽다.)

# 코너의 특징 
# - 평탄한 영역(flat), 에지(edge)영역은 고유한 위치를 찾기 힘들다.
# - 코너는 변별력이 높으며 영상의 이동, 회전 변환에 강하다.

import sys
import numpy as np
import cv2


src = cv2.imread('.\AR_video_player.\\building.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

tm = cv2.TickMeter()

# GFTT
tm.start()

#추적하기 좋은 특징 코너 검출 함수
corners = cv2.goodFeaturesToTrack(src, 400, 0.01, 10)

tm.stop()
print('GFTT: {}ms.'.format(tm.getTimeMilli()))

dst1 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

if corners is not None:
    for i in range(corners.shape[0]): #코너의 개수만큼
        pt = (int(corners[i, 0, 0]), int(corners[i, 0, 1])) #점의 좌표를 int형으로 변환
        cv2.circle(dst1, pt, 5, (0, 0, 255), 2) #코너에 원을 그린다.




# FAST
# Fast 방법이 반복 검출률이 높고 속도가 빠르다 
# but 노이즈에 민감하다.
tm.reset()
tm.start()

fast = cv2.FastFeatureDetector_create(60) #60은 threshold로 중앙 값과 60이상 차이나는 것을 찾기 위함
keypoints = fast.detect(src) 

tm.stop()
print('FAST: {}ms.'.format(tm.getTimeMilli()))

dst2 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

for kp in keypoints:
    pt = (int(kp.pt[0]), int(kp.pt[1])) #점의 좌표를 int형으로 변환
    cv2.circle(dst2, pt, 5, (0, 0, 255), 2) #코너에 원을 그린다.

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()

cv2.destroyAllWindows()
