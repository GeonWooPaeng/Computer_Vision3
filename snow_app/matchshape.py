# 모멘트 
# 영상의 형태(모양정보)를 표현하는 일련의 실수 값(특징 벡터를 추출하는 것) 

# Hu의 7개 불변 모멘트(Hu's seven invariant moments)방법
# Normalized central moments를 조합하는 방법 -> 두 외곽선 또는 영상의 모양을 비교
# 영상의 크기, 회전, 이동, 대칭 변환에 비교를 잘 할 수 있다.
# 찌그러짐 등에는 잘 비교하지 못한다 .


# 원하는 그림을 원하는 그림이 있는 영상에서 찾아내는 프로그램 

import sys
import numpy as np
import cv2


# 영상 불러오기
obj = cv2.imread('.\snow_app.\spades.png', cv2.IMREAD_GRAYSCALE)
src = cv2.imread('.\snow_app.\symbols.png', cv2.IMREAD_GRAYSCALE)

if src is None or obj is None:
    print('Image load failed!')
    sys.exit()

# 객체 영상 외곽선 검출
_, obj_bin = cv2.threshold(obj, 128, 255, cv2.THRESH_BINARY_INV)
obj_contours, _ = cv2.findContours(obj_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #외곽선 검출
obj_pts = obj_contours[0]

# 입력 영상 분석
_, src_bin = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 결과 영상
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

# 입력 영상의 모든 객체 영역에 대해서
for pts in contours: #pts: 외곽선 하나의 정보
    if cv2.contourArea(pts) < 1000: #노이즈를 제거하기 위함
        continue

    rc = cv2.boundingRect(pts)
    cv2.rectangle(dst, rc, (255, 0, 0), 1)

    # 모양 비교
    dist = cv2.matchShapes(obj_pts, pts, cv2.CONTOURS_MATCH_I3, 0) #모양 비교 함수(값으로 비교한다.)

    cv2.putText(dst, str(round(dist, 4)), (rc[0], rc[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

    if dist < 0.1:
        cv2.rectangle(dst, rc, (0, 0, 255), 2)

cv2.imshow('obj', obj)
cv2.imshow('dst', dst)
cv2.waitKey(0)

