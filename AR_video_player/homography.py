# 호모그래피
# 두 평면 사이의 투시 변환(3x3)
# 원래 사진과 비스듬한 사진을 비교한 것

import sys
import numpy as np
import cv2


# 영상 불러오기
# src1 = cv2.imread('.\AR_video_player.\graf1.png', cv2.IMREAD_GRAYSCALE)
# src2 = cv2.imread('.\AR_video_player.\graf3.png', cv2.IMREAD_GRAYSCALE)
src1 = cv2.imread('.\AR_video_player.\\box.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('.\AR_video_player.\\box_in_scene.png', cv2.IMREAD_GRAYSCALE)

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

# 특징점 매칭
matcher = cv2.BFMatcher_create()
#matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
matches = matcher.match(desc1, desc2)

# 좋은 매칭 결과 선별(상위 80개만 사용)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:80]

print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of matches:', len(matches))
print('# of good_matches:', len(good_matches))

# 호모그래피 계산
# queryIdx: 1번 img에서의 keypoint 번호 / .pt 점의 좌표
pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2).astype(np.float32)

# trainIdx: 2번 img에서의 keypoint 번호 / .pt 점의 좌표 
pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2).astype(np.float32)

# H에는 두 영상관의 perspective 관계를 가지고 있다.
H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC) #호모그래피 계산 함수

# 호모그래피를 이용하여 기준 영상 영역 표시(박스로 표현)
dst = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

(h, w) = src1.shape[:2]
corners1 = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                    ).reshape(-1, 1, 2).astype(np.float32) #입력 영상
corners2 = cv2.perspectiveTransform(corners1, H) # 점을 가지고 투시변환 행렬(H)을 가지고 이동 시키는 것
corners2 = corners2 + np.float32([w, 0]) # shift 해주는 곳

cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
