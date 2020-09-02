# 이미지 스티칭
# 여러장의 사진을 자연스럽게 붙여서 한장의 사진으로 만드는 기술 
# 사진 이어 붙이기, 파노라마 영상

# 방법
# 여러장의 영상에서 특징점을 검출하고 특징점이 겹치는 것을 찾아서 두 영상의 perspective부분을 찾아서 이어 붙인다.

import sys
import numpy as np
import cv2


img_names = ['.\AR_video_player.\img1.jpg', '.\AR_video_player.\img2.jpg', '.\AR_video_player.\img3.jpg']

imgs = []
for name in img_names:
    img = cv2.imread(name)

    if img is None:
        print('Image load failed!')
        sys.exit()

    imgs.append(img)

stitcher = cv2.Stitcher_create() # 이미지 스티칭 객체 생성
status, dst = stitcher.stitch(imgs) # 이미지 스티칭 함수(영상 합성)

if status != cv2.Stitcher_OK: 
    print('Stitch failed!')
    sys.exit()

cv2.imwrite('output.jpg', dst) # 결과 저장

cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
