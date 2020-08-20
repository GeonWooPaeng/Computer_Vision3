# 레이블링 
# 키보드에서 특정 문자를 찾아내기 

import sys
import numpy as np
import cv2


src = cv2.imread('.\Business_card_recognition_program.\keyboard.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# src_bin - 이진영상
_, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU) #이진화


# cnt는 배경까지 포함해서 +1 해준 값이 나온다.
# stats: 어느 위치에 어느정도 크기로 존재하냐(각 객체의 바운딩 박스), 픽셀 개수 정보를 담은 행렬
# - numpy.ndarray.shape = (N,5), dtype=numpy.int32 
# centroids: 각 객체의 무게 중심 위치 정보를 담은 행렬
# - numpy.ndarray.shape = (N,2), dtype=numpy.float64
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin) #레이블링


dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

#객체 위치에 다가 빨간색 사각형을 그리기
for i in range(1, cnt): #1부터시작 이유: 배경을 제외하기 때문
    (x, y, w, h, area) = stats[i]

    # 작은 점들 무시하기 위함 
    if area < 20:
        continue

    cv2.rectangle(dst, (x, y, w, h), (0, 0, 255), 2)

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
