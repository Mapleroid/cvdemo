import cv2
import numpy as np

cap=cv2.VideoCapture(0)

def img2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def thresholdimg2binary(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(gray_img,128,255, cv2.THRESH_BINARY)

    return binary_img

def otsuimg2binary(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(gray_img,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return binary_img

def adaptiveThresholdimg2binary(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(
            gray_img,
            255,                    # Value to assign
            cv2.ADAPTIVE_THRESH_MEAN_C,# Mean threshold
            cv2.THRESH_BINARY,
            11,                     # Block size of small area
            2,                      # Const to substract
        )
    return binary_img

def canny_edge(img, min=100, max=200):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img,(3,3),0)
    return cv2.Canny(blur, min, max)

while(1):
    ret,frame=cap.read()
    if not ret:
        continue

    # to gray image
    #gray_img = img2gray(frame)

    # to binary image by threshold
    #binary_img = otsuimg2binary(frame)

    # to binary image by adaptiveThreshold
    binary_img = adaptiveThresholdimg2binary(frame)
    
    # to binary image by Canny
    #edge_img = canny_edge(frame)

    #origin = frame.copy()

    contours_img, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter
    contour_m = np.array([[[0,0],[0,100],[100,100],[100,0]]])

    list = hierarchy[0]
    contour_idxs = []
    for index in xrange(len(list)):
        level=1
        child = list[index][2]
        while child!=-1:
            level+=1
            child=list[child][2]

        if level>=3 and level<=5 and cv2.matchShapes(contour_m,contours[index],1,0.0)<0.05:
            contour_idxs.append(index)

    colors = ( (0,0,255), (255,0,0), (0,255,0), (255,255,0),(0,255,255))

    # check
    #bimg = np.zeros((480,640,3),dtype=np.int32)
    #bimg[:,:,2]
    bimg = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    patterns = []
    for index in contour_idxs:
        son = list[index][2]
        grandson = list[son][2]

        area_self = cv2.contourArea(contours[index])
        area_son = cv2.contourArea(contours[son])
        area_grandson = cv2.contourArea(contours[grandson])

        ratio_1 = area_son/area_self
        ratio_2 = area_grandson/area_son
        ratio_3 = area_grandson/area_self

        if (ratio_1>0.25 and ratio_1<0.75) and (ratio_2>0.18 and ratio_2<0.54) and (ratio_3>0.09 and ratio_3<0.27):
            patterns.append(index)
            print (ratio_1,ratio_2,ratio_3)
            #cv2.drawContours(bimg,contours, index,(0,0,255), 2)
            #level=1
            #child = list[index][2]
            #while child!=-1:
            #    level += 1
            #    cv2.drawContours(bimg,contours, child,colors[level-1], 2)
            #    child=list[child][2]


    if 3<=len(patterns):
        for index in patterns:
            cv2.drawContours(bimg,contours, index,(0,0,255), 1)
        #    x, y, w, h = cv.boundingRect(patterns[ind])
        #print index
        #print (area_son/area_self, area_grandson/area_son) 
        #cv2.drawContours(frame,contours, index,(0,255,0), 1)

        #leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        #rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        #topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        #bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        #
        #cv2.line(frame,leftmost,bottommost,(255,0,0),1)
        #cv2.line(frame,bottommost,rightmost,(255,0,0),1)
        #cv2.line(frame,rightmost,topmost,(255,0,0),1)
        #cv2.line(frame,topmost,leftmost,(255,0,0),1)
        #
        #cv2.drawContours(frame,contours, index,(0,255,0), 1)
        #child = list[index][2]
        #while child!=-1:
        #    cv2.drawContours(frame,contours, child,(0,255,0), 1)
        #    child=list[child][2]

    #for index in xrange(len(contours)):
    #    if cv2.contourArea(contours[index])>48:
    #        cv2.drawContours(frame,contours, index,(0,255,0), 1)

    #cv2.drawContours(origin,contours, -1,(0,255,0), 1)
  
    # end process
    cv2.imshow('res1',bimg)
    #cv2.imshow('res2',origin)

    k=cv2.waitKey(5)&0xFF
    if k==27:
        break

cv2.destroyAllWindows()