# 모폴로지 연산
# 영상을 형태학적으로 처리한다.

# 이진 영상의 열기(opening) 연산
# 열기 = 침식(erosion) 후 팽창(dilation)한 것 
# 작은 크기 객체 삭제, 얇은 선이 끊어진다.

# 이진 영상의 닫기(closing) 연산
# 닫기 = 팽창 후 침식한 것
# 작은 홀이 매꿔지고 얇은 선은 두꺼워진다.

import sys
import numpy as np
import cv2


src = cv2.imread('.\Business_card_recognition_program.\\rice.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# src 영상에 지역 이진화 수행 (local_th.py 참고)
dst1 = np.zeros(src.shape, np.uint8)

bw = src.shape[1] // 4
bh = src.shape[0] // 4

for y in range(4):
    for x in range(4):
        src_ = src[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        dst_ = dst1[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
        cv2.threshold(src_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst_)

cnt1, _ = cv2.connectedComponents(dst1) # 영상안에 있는 흰색 개수를 정수형태로 return 해준다.
print('cnt1:', cnt1)

#mophologyEx()은 침식, 팽창, 열기, 닫기 다 수행가능하다
dst2 = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, None)
#dst2 = cv2.erode(dst1, None)
#dst2 = cv2.dilate(dst2, None)

cnt2, _ = cv2.connectedComponents(dst2)
print('cnt2:', cnt2)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
