# 특징점 매칭
# 두 영상에서 각각 특징점 근방에서 특징점 기술자를 추출하여 
# 두 영상에서 비슷한 기술자를 찾아 매칭하는 것

# 특징 벡터 유사도 측정 
# 실수 특징 벡터: L2 노름(L2 norm)사용
# 이진 특징 벡터: 해밍 거리(hamming distance)사용

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
#feature = cv2.AKAZE_create()
#feature = cv2.ORB_create()

# 특징점 검출 및 기술자 계산
kp1, desc1 = feature.detectAndCompute(src1, None)
kp2, desc2 = feature.detectAndCompute(src2, None)

# 특징점 매칭 알고리즘 객체 생성
matcher = cv2.BFMatcher_create() # 실수 기술자
#matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING) #이진 기술자
matches = matcher.match(desc1, desc2) #전역 검사

print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of matches:', len(matches))

# 특징점 매칭 결과 영상 생성(매칭된 결과를 특징점 끼리 연결한 것을 보여준다.)
dst = cv2.drawMatches(src1, kp1, src2, kp2, matches, None)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
