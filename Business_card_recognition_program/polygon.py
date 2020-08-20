# 다양한 외곽선 함수
# 도형사진에서 사각형, 삼각형, 원등을 찾는 프로그램 

# 구현 순서 
#1. 이진화
#2. 외곽선 찾기
#3. 외곽선 근사화(점으로 근사화)
#4. 너무 작은 객체와 convex가 아닌 객체 제외
#5. 꼭지점(점) 개수 확인 -> 판별


# 원 판별
# 도형의 넓이(A)와 외곽선 길이(P)의 비율을 검사
# -> 정해진 외곽선 길이에 대한 넓이 비율이 가장 큰 형태가 원


import math
import cv2

# 주요 함수
#cv2.arcLength() - 외곽선 길이 구하기 
#cv2.contourArea() - 외곽선 면적 구하기
#cv2.boundingRect() - 바운딩 박스(외곽선을 외접하여 둘러싸는 가장 작은 사각형) 구하기
#cv2.minEnclosingCircle() - 바운딩 서클(욍곽선을 외접하여 둘러싸는 가장 작은 원) 구하기
#cv2.approxPolyDP() - 외곽선 근사화(단순화) -> 출력 k값이 줄어들 수 있다.
#cv2.isContourConvex - Convex검사(볼록이냐 오목같이 들어가있나 검사)

def setLabel(img, pts, label):
    # 좌표계산해서 label문자열 출력
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))


def main():
    img = cv2.imread('.\Business_card_recognition_program.\polygon.bmp', cv2.IMREAD_COLOR)

    if img is None:
        print('Image load failed!')
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale로 변화
    _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) #threshold 한다, THRESH_BINARY_INV - 도형이 더 어두운 형태이다.
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #각각의 객체를 찾아낸다.(외곽선 검출)

    for pts in contours:

        #noise를 제거하기 위한 코드
        if cv2.contourArea(pts) < 400:  #  너무 작으면 무시
            continue


        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02, True) #외곽선 근사화

        vtc = len(approx)

        if vtc == 3:
            #삼각형
            setLabel(img, pts, 'TRI')

        elif vtc == 4:
            #사각형
            setLabel(img, pts, 'RECT')

        else:
            # 원 판별 부분
            length = cv2.arcLength(pts, True)
            area = cv2.contourArea(pts)
            ratio = 4. * math.pi * area / (length * length)

            if ratio > 0.85:
                setLabel(img, pts, 'CIR')

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
