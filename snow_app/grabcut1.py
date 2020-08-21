# 그랩컷 
# - 그래프 컷 기반 영역 분할 알고리즘 (배경과 객체를 구분해준다.)
# - 영상의 픽셀을 그래프 정점으로 간주하고, 
# 픽셀들을 두 개의 그룹으로 나누는 최적의 컷(Max Flow Minimum Cut)을 찾는 방식

#사용자 지정(ROI)으로 그랩컷 하는 것

import sys
import numpy as np
import cv2


# 입력 영상 불러오기
src = cv2.imread('.\snow_app.\\nemo.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

# 사각형 지정을 통한 초기 분할
rc = cv2.selectROI(src) #ROI 창이 뜬다(사각형 사용자 지정하기 위함)

mask = np.zeros(src.shape[:2], np.uint8)

cv2.grabCut(src, mask, rc, None, None, 5, cv2.GC_INIT_WITH_RECT) #그랩컷 하는 부분 (mask 계속 update한다.)

# 0: cv2.GC_BGD(), 2: cv2.GC_PR_BGD() -> background 인것 or 인것같은 것
# 1: cv2.GC_FGD(), 3: cv2.GC_PR_FGD() -> 객체 인것 or 인것 같은것
mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]

mask = mask * 64

# 초기 분할 결과 출력
cv2.imshow('dst', dst)
cv2.imshow('mask',mask)
cv2.waitKey()
cv2.destroyAllWindows()
