# 모폴로지 연산
# 영상을 형태학적으로 처리한다.

# 1. 이진 영상의 침식(erosion) 연산
# 객체 영영(객체 외각)이 깎여나가는 연산 -> 객체 크기는 감소 & 배경은 확대 
# 객체 영역(흰색)이 점점 줄어든다.
# 작은 크기의 객체(잡음) 제거
# 'structuring element가 객체 안에 모두 포함'될 때 가운데 부분을 marking을 하고 
# marking된 부분만 남기고 나머지 객체는 지운다.

# 2. 이진 영상의 팽창(dilation) 연산
# 객체 영역(객체 외각)을 확대시키는 연산 -> 객체 크기 감소 & 배경 확대
# 객체 영역(흰색)이 점점 늘어난다.
# 객체 내부의 홀(구멍)이 채워짐.
# 'structuring element가 객체랑 하나라도 포함'되면 가운데 부분을 marking을 하고 
# marking된 부분만 남기고 나머지 객체는 지운다.


import sys
import numpy as np
import cv2

# 이진화 되어있는 영상
src = cv2.imread('.\Business_card_recognition_program.\circuit.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)) #erode()함수에서 사용할 kernel 만들기(structuring element만들기)
dst1 = cv2.erode(src, se) #수축

dst2 = cv2.dilate(src, None) #팽창, kernel을 None으로 주면 3x3 사각형이다.

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
