# 기술자(descriptor, feature vector)
# 특징점 근처의 부분영상을 표현하는 실수 or 이진벡터

# 실수 기술자(L2 노름(L2 norm) 사용)
# 주로 특징점 근처 부분 영상의 방향 히스토그램 사용
# 알고리즘: SIFt, SURF, KAZE

# 이진 기술자(해밍 거리)
# 이진 테스트를 이용해서 부분 영상의 특징을 기술(밝기 차이를 이진수로 표현)
# 알고리즘: AKAZE, ORB, BRIEF 

import sys
import numpy as np
import cv2


# 영상 불러오기
src1 = cv2.imread('.\AR_video_player.\graf1.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('.\AR_video_player.\graf3.png', cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

# 특징점 알고리즘 객체 생성 (KAZE, AKAZE, ORB 등)
# feature = cv2.KAZE_create()
#feature = cv2.AKAZE_create()
feature = cv2.ORB_create()

# 특징점 검출 및 기술자 계산
kp1 = feature.detect(src1)
_, desc1 = feature.compute(src1, kp1) # 특징점 기술자 계산 함수

kp2, desc2 = feature.detectAndCompute(src2, None) #특징점 검출 + 기술자 계산 함수

print('desc1.shape:', desc1.shape)
print('desc1.dtype:', desc1.dtype)
print('desc2.shape:', desc2.shape)
print('desc2.dtype:', desc2.dtype)

# 검출된 특징점 출력 영상 생성
dst1 = cv2.drawKeypoints(src1, kp1, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst2 = cv2.drawKeypoints(src2, kp2, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
