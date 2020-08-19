# 영상의 이진화
# 영상의 픽셀 값을 0또는 255(1)로 만드는 연산
# 임계값 = 문턱치 = threshold => 나누기 위해 기준이 되는 값

import sys
import numpy as np
import cv2


src = cv2.imread('.\Business_card_recognition_program.\cells.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# 임계값을 이용하여 할 수 있는 함수 (완전히 이진화하지는 못한다)
# retval, dst를 반환한다.
_, dst1 = cv2.threshold(src, 100, 255, cv2.THRESH_BINARY)
_, dst2 = cv2.threshold(src, 210, 255, cv2.THRESH_BINARY)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()


