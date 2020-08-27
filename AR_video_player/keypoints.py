# 특징점 검출
# 특징점 검출 -> 매칭 -> 같은 그림 파악 

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
feature = cv2.KAZE_create()
#feature = cv2.AKAZE_create() #방향성분도 나온다.
#feature = cv2.ORB_create()


# 특징점 검출
# keyPoint 
# - pt: 특징점의 x,y좌표를 float형태로 표현
# - size: 특징점을 검출할때 어느정도 크기를 가지고 검출하였나
# - angle: 부분 영상의 주된 방향
kp1 = feature.detect(src1)
kp2 = feature.detect(src2)

print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))

# 검출된 특징점 출력 영상 생성(검출된 특징점 그리기 함수)
dst1 = cv2.drawKeypoints(src1, kp1, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst2 = cv2.drawKeypoints(src2, kp2, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
