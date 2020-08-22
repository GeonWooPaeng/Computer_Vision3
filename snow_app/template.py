# 템플릿 매칭
# 입력영상에서 (작은 크기의)템플릿 영상과 일치하는 부분을 찾는 기법
# 템플릿(커널): 찾을 대상이 되는 작은 영상. 패치(patch)
# 밝기 차이가 있는 영상은 잘찾지만 크기, 회전에 대해서는 잘 찾지 못한다.
 
#순서
# 템플릿 스캔(전체) -> 유사도/비유사도 -> 최댓값(유사도)/최솟값(비유사도) 선택 -> 템플릿 매칭
import sys
import numpy as np
import cv2


# 입력 영상 & 템플릿 영상 불러오기
src = cv2.imread('.\snow_app.\circuit.bmp', cv2.IMREAD_GRAYSCALE)
templ = cv2.imread('.\snow_app.\crystal.bmp', cv2.IMREAD_GRAYSCALE)

if src is None or templ is None:
    print('Image load failed!')
    sys.exit()

# 입력 영상 밝기 50증가, 가우시안 잡음(sigma=10) 추가
noise = np.zeros(src.shape, np.int32)
cv2.randn(noise, 50, 10) #noise영상(행렬)에 평균 50이고 sigma값이 10인 형태의 가우시안 노이즈 추가
src = cv2.add(src, noise, dtype=cv2.CV_8UC3)

# 템플릿 매칭 & 결과 분석
res = cv2.matchTemplate(src, templ, cv2.TM_CCOEFF_NORMED) #템플릿 매칭 함수
res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) #res라는 실수형 행렬을 grayscale로 보는 것(범위 0 ~255)

_, maxv, _, maxloc = cv2.minMaxLoc(res) #최댓값 위치를 찾아야 하기 때문
print('maxv:', maxv) #해당 값으로 0.7정도 이상되어야 해당 영상이 있는 것이다.
print('maxloc:', maxloc)

# 매칭 결과를 빨간색 사각형으로 표시
th, tw = templ.shape[:2]
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
cv2.rectangle(dst, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)

# 결과 영상 화면 출력
cv2.imshow('res_norm', res_norm) #출력형태를 grayscale 형태로 보여준다.
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
