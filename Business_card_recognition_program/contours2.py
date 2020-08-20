import sys
import random
import numpy as np
import cv2


src = cv2.imread('.\Business_card_recognition_program.\milkdrop.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# _ 은 Otsu방법에의해 결정된 threshold값
# src_bin 은 이진 영상
_, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU) #이진화

# _ 은 hierarchy
# contours의 길이 = 검출한 외곽선의 개수
contours, _ = cv2.findContours(src_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #외곽선 검출

h, w = src.shape[:2]
dst = np.zeros((h, w, 3), np.uint8)

for i in range(len(contours)):
    #이어져 있는 외곽선은 같은 색이 나온다.
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst, contours, i, c, 1, cv2.LINE_AA) #외곽선을 보여주기 위함

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
