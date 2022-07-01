
import cv2
import numpy as np
import time

def ShowRoi(ok_img, ok_roi,w,h):
    roi = ok_roi.copy()
    cv2.rectangle(ok_roi, (0,0),(h-1,w-1),(0,255,0))
    cv2.imshow('draw roi',ok_img)
    cv2.namedWindow('ok roi',cv2.WINDOW_NORMAL)
    cv2.imshow('ok roi',roi)
    return roi

def RotateImg(img, dx):
    rows, cols = img.shape[0:2]
    mdx = cv2.getRotationMatrix2D((cols/2,rows/2),dx,1)
    img = cv2.warpAffine(img,mdx,(cols,rows))
    return img

def TemplateMatching(ok_roi, target_img, method):
    start = time.time()
    res = cv2.matchTemplate(target_img, ok_roi, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val
    bottom_right = (top_left[0] + w_1, top_left[1] + h_1)
    # print(match_val)
    print("time :", time.time() - start) 
    return top_left, bottom_right, match_val


test_ok_img = cv2.imread('./img/0.jpg')
first_ok_img = cv2.imread('./img/1(1000_225).jpg')
second_ok_img = cv2.imread('./img/9(1500_225).jpg')
third_ok_img = cv2.imread('./img/30(1500_255).jpg')
fourth_ok_img = cv2.imread('./img/36(1500_255).jpg')
target_img = cv2.imread('./img/40(1500_255).jpg')
draw_img = target_img.copy()

x = 200;y = 350;w = 100;h = 100
test_ok_roi = test_ok_img[y:y+h, x:x+w]
#test_ok_roi = ShowRoi(test_ok_img, test_ok_roi, w, h)
test_ok_roi = test_ok_roi.copy()

x_1 = 117; y_1 = 370; w_1 = 125; h_1 = 110
first_ok_roi = first_ok_img[y_1:y_1+h_1, x_1:x_1+w_1]
#first_ok_roi = ShowRoi(first_ok_img, first_ok_roi, w_1, h_1)
first_ok_roi = first_ok_roi.copy()

x_2 = 165;y_2 = 320;w_2 = 165;h_2 = 120
second_ok_roi = second_ok_img[y_2:y_2+h_2, x_2:x_2+w_2]
#second_ok_roi = ShowRoi(second_ok_img, second_ok_roi,w_2,h_2)
second_ok_roi = second_ok_roi.copy()

x_3 = 210; y_3 = 450; w_3 = 110;h_3 = 110
third_ok_roi = third_ok_img[y_3:y_3+h_3, x_3:x_3+w_3]
#third_ok_roi = ShowRoi(third_ok_img, third_ok_roi, w_3, h_3)
third_ok_roi = third_ok_roi.copy()

x_4 = 230;y_4 = 460;w_4 = 100;h_4 = 100
fourth_ok_roi = fourth_ok_img[y_4:y_4+h_4, x_4:x_4+w_4]
#fourth_ok_roi = ShowRoi(fourth_ok_img, fourth_ok_roi,w_4,h_4)
fourth_ok_roi = fourth_ok_roi.copy()

#cv2.TM_SQDIFF:제곱 차이 매칭, 완벽 매칭:0,나쁜 매칭:큰값/
#cv2.TM_SQDIFF_NORMED:제곱 차이 매칭의 정규화
#cv2.TM_CCORR:상관관계 매칭,완벽 매칭:큰값, 나쁜 매칭:0/
#cv2.TM_CCORR_NORMED:상관관계 매칭의 정규화
#cv2.TM_CCOEFF:상관계수 매칭, 완벽 매칭:1,나쁜매칭:-1/
#cv2.TM_CCOEFF_NORMED:상관계수의 매칭 정규화
top_left, bottom_right, match_val = TemplateMatching(fourth_ok_roi, target_img, cv2.TM_CCOEFF_NORMED)

cv2.rectangle(draw_img, top_left, bottom_right, (0,255,0))
cv2.putText(draw_img, str(match_val), top_left, cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1,cv2.LINE_AA)

cv2.imwrite('./matchImg/40(1500_255).jpg',draw_img)
cv2.imshow('target',draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
