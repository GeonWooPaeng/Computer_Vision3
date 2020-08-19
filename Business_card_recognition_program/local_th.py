# 불균일한 조명성분에서 사용하는 지역 이진화

# 해결방안
# 1. 전역 이진화
# 불균일한 조명(surface fitting) 성분을 보정(shading compensation)한 후 전역 이진화 수행(global thresholding)

# 2. 지역 이진화 
# 전체 영상을 n등분 한 후 각각 구역마다 이진화 하고 이어 붙이는 형태

import sys
import numpy as np
import cv2


# 입력 영상 불러오기
src = cv2.imread('.\Business_card_recognition_program.\\rice.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# 전역 이진화 by Otsu's method
_, dst1 = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 지역 이진화 by Otsu's method
dst2 = np.zeros(src.shape, np.uint8) #검정색으로 채워진 영상

bw = src.shape[1] // 4 # 512 // 4 = 128
bh = src.shape[0] // 4

for y in range(4):
    for x in range(4):
        src_ = src[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        dst_ = dst2[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        #src_ 와 dst_ 크기가 같아야 한다.
        cv2.threshold(src_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst_) #dst_는 입력이자 출력인 것

# 결과 출력
cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
