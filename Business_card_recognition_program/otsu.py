# 자동 이진화
# 임계값 자동 결정 방법
# 픽셀 분포 비슷o -> 픽셀 값 평균(임계값)
# 픽셀 분포 비슷x -> Otsu 방법

# Otsu 이진화 방법
# 배경 & 객체 두개로 구성되어 있다고 가정(histogram 전체 모양이 Bimodel(산 모양) 인 경우 적용) 
# 임의의 임계값(T)을 정해서 pixel그룹의 T보다 큰쪽 분산 T보다 작은쪽 분산 계산해서 분산의 값이 최소가 되는 T값을 찾는 방식
# => 분산이 작다(많이 모여있다.)

import sys
import numpy as np
import cv2


src = cv2.imread('.\Business_card_recognition_program.\\rice.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

#Otsu 방법을 사용하려면 min값에 0을 줘야된다.
th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)# 위 코드랑 같다
# th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)# 반전해서 Otsu 이진화
print("otsu's threshold:", th)  # 131(임계값)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()



