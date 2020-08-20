# 객체 단위 분석
# labeling 기법 (Connected component labeling)
# 이진영상에서 객체의 모양과 크기를 추출하는 기법 
# 객체 구역을 영역 단위로 분석하는 방법 
# 같은 객체에 같은 고유한 번호를 매기는 기법 

# 연결성을 미리 정의해야 한다.
# 4-neightbor: 상하좌우로 연결되어 있을 때 
# 8-neightbor: 4-neightbor + 대각선

import sys
import numpy as np
import cv2


mat = np.array([
    [0, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)

cnt, labels = cv2.connectedComponents(mat)

print('sep:', mat, sep='\n')
print('cnt:', cnt)
print('labels:', labels, sep='\n')
