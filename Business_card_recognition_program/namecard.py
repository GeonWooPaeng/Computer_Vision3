"""
Tesseract-ocr 설치하기

1. tesseract-ocr-w64-setup-v5.0.0-alpha.20200328 파일 다운로드 
   (https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20200328.exe)
2. 설치 시 "Additional script data" 항목에서 "Hangul Script", "Hangul vertical script" 항목 체크,
   "Additional language data" 항목에서 "Korean" 항목 체크.
4. 설치 후 시스템 환경변수 PATH에 Tesseract 설치 폴더 추가
   (e.g.) c:\Program Files\Tesseract-OCR
4. 설치 후 시스템 환경변수에 TESSDATA_PREFIX를 추가하고, 변수 값을 <Tesseract-DIR>\tessdata 로 설정
5. <Tesseract-DIR>\tessdata\script\ 폴더에 있는 Hangul.traineddata, Hangul_vert.traineddata 파일을
   <Tesseract-DIR>\tessdata\ 폴더로 복사
6. 명령 프롬프트 창에서 pip install pytesseract 명령 입력
"""

# 명함 인식 프로그램 
# 영상에서 명함을 검출하고, 명함 안의 텍스트를 인식하는 프로그램 
# 가정 - 명함은 흰색이고, 충분히 크게 촬영되었고, 각진 사각형 모양이다.

# 순서
# 이진화 -> 외곽선 검출 & 다각형 근사화 -> 투시변환 -> OCR

#명함 검출 
# Otsu 자동 이진화 
# 외곽선 검출 -> 근사화 -> 사각형 검출 
# 명함의 네모서리 점을 직사각형 네 모서리로 매핑 
# - 좌측 상단 모서리점부터 반시계방향으로 검출 및 매핑
# 일반적인 명함의 가로:세로 비율 = 9:4

import sys
import random
import numpy as np
import cv2
import pytesseract


def reorderPts(pts):
    # 외곽선 근사화 점 재정의 
    #1. 점들의 좌표를 x좌표 기준으로 2개씩 나눈다.
    #2. 나눈 2개를 가지고 y좌표를 기준으로 나눈다.
    # x 작은거 중 y작은거 0번 y큰거 1번 x큰거 중 y작은거 2번 y큰거 3번
    idx = np.lexsort((pts[:, 1], pts[:, 0]))  # 칼럼0 -> 칼럼1 순으로 정렬한 인덱스를 반환
    pts = pts[idx]  # x좌표로 정렬

    if pts[0, 1] > pts[1, 1]: # y좌표 비교 하여 작은 놈을 0번으로
        pts[[0, 1]] = pts[[1, 0]]

    if pts[2, 1] < pts[3, 1]: 
        pts[[2, 3]] = pts[[3, 2]]

    return pts


# 영상 불러오기
filename = '.\Business_card_recognition_program.\\namecard1.jpg'
if len(sys.argv) > 1:
    filename = sys.argv[1]

src = cv2.imread(filename)

if src is None:
    print('Image load failed!')
    sys.exit()

# 출력 영상 설정
dw, dh = 720, 400 #출력 영상 크기
srcQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32) #입력영상에서 명함의 네개 모서리부분을 저장할 부분
dstQuad = np.array([[0, 0], [0, dh], [dw, dh], [dw, 0]], np.float32) #dw,dh 직사각형을 좌측 상단부터 반시계방향으로 도는 형태로 저장
dst = np.zeros((dh, dw), np.uint8) #출력

# 입력 영상 전처리
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)# gray_scale변환
_, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)# 이진화

# 외곽선 검출 및 명함 검출
contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cpy = src.copy()
for pts in contours: #각각의 외곽선 정보를 pts로 받게 해준다.
    # 너무 작은 객체는 무시
    if cv2.contourArea(pts) < 1000:
        continue

    # 외곽선 근사화(점 몇개로 근사화 됬는지 확인)
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02, True)

    # 컨벡스가 아니고, 사각형이 아니면 무시
    if not cv2.isContourConvex(approx) or len(approx) != 4:
        continue

    cv2.polylines(cpy, [approx], True, (0, 255, 0), 2, cv2.LINE_AA) #외곽선 그리는 작업
    srcQuad = reorderPts(approx.reshape(4, 2).astype(np.float32)) #외곽선 근사화에서 점의 순서를 재정의 해주는 곳

pers = cv2.getPerspectiveTransform(srcQuad, dstQuad) #투시변환 행렬구하기
dst = cv2.warpPerspective(src, pers, (dw, dh)) # 펴는 작업

dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) 
print(pytesseract.image_to_string(dst_gray, lang='Hangul+eng')) #lang='Hangul+eng'한글과 영어를 인식해라

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
