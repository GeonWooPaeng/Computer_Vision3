# 불균일한 조명성분에서 사용하는 지역 이진화

# 지역 이진화중 적응형 이진화 
# src를 블러링을 한다.
# 속도가 느리다.

# 스도쿠 사각형 안에 검정색이 나타나는 이유
# blocksize가 작을 경우 block안에 배경 or 객체만 있는 경우 오동작을 하기 때문

import sys
import numpy as np
import cv2


src = cv2.imread('.\Business_card_recognition_program.\sudoku.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()


def on_trackbar(pos):
    bsize = pos
    if bsize % 2 == 0:
        bsize = bsize - 1
    if bsize < 3:
        bsize = 3

    dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, bsize, 5)

    cv2.imshow('dst', dst)


cv2.imshow('src', src)
cv2.namedWindow('dst')
cv2.createTrackbar('Block Size', 'dst', 0, 200, on_trackbar)
cv2.setTrackbarPos('Block Size', 'dst', 11)

cv2.waitKey()
cv2.destroyAllWindows()
