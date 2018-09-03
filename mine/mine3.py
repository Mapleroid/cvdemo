import time
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import ctypes
import win32gui, win32ui, win32con, win32api

import utils

def find_mine():
    wndtitle = None
    wndclass = "Minesweeper"
    wnd = win32gui.FindWindow(wndclass, wndtitle)
    return wnd

def set_dpi_aware():
    try:
        f = ctypes.windll.shcore.SetProcessDpiAwareness
        PROCESS_SYSTEM_DPI_AWARE = 1
        f(PROCESS_SYSTEM_DPI_AWARE)
        return True
    except WindowsError:
        return False       

def get_client_rect(hwnd):
    l, t, r, b = win32gui.GetClientRect(hwnd)
    w = r-l
    h = b-t

    return (l, t, w, h)
    
def adaptiveThresholdimg2binary(gray_img):
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(
            gray_img,
            255,                    # Value to assign
            cv2.ADAPTIVE_THRESH_MEAN_C,# Mean threshold
            cv2.THRESH_BINARY,
            11,                     # Block size of small area
            2,                      # Const to substract
        )
    return binary_img

def get_mine_area(rgb_shot):
    gray_img = cv2.cvtColor(rgb_shot, cv2.COLOR_RGB2GRAY)
    edge_img = adaptiveThresholdimg2binary(gray_img)
    contours_img, contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_idxs = []
    for index in xrange(len(contours)):
        arcl = cv2.arcLength(contours[index],True)
        if arcl > (rgb_shot.shape[0]+rgb_shot.shape[1])*2*0.8:
            contour_idxs.append(index)

    if len(contour_idxs)!=2:
        return None
    
    area1 = cv2.contourArea(contours[contour_idxs[0]])
    area2 = cv2.contourArea(contours[contour_idxs[1]])

    if area1 > area2:
        return cv2.boundingRect(contours[contour_idxs[1]])
    else:
        return cv2.boundingRect(contours[contour_idxs[0]])

def get_shot(bmp, mem_dc, client_dc, l, t, w, h):
    # copy from client dc to memory dc
    mem_dc.BitBlt((0, 0), (w, h), client_dc, (l, t), win32con.SRCCOPY)
    # save memory dc's content to bitmap
    #bmp.SaveBitmapFile(mem_dc, 'screenshot.bmp')
    img_shot = bmp.GetBitmapBits()

    bmparray = np.asarray(img_shot, dtype=np.uint8)
    pil_im = Image.frombuffer('RGB', (w, h), bmparray, 'raw', 'BGRX', 0, 1)
    rgb = np.array(pil_im)
    return rgb

def get_img_at_pos(img_shot, mine_area, x, y):
    area_x, area_y, area_w, area_h = mine_area
    
    box_w = 1.0*area_w/30
    box_h = 1.0*area_h/16

    box_x = int(area_x + x*box_w)
    box_y = int(area_y + y*box_h)

    return img_shot[box_y:(box_y+int(box_h)), box_x:(box_x+int(box_w))]
    #return (box_x, box_y, int(box_w), int(box_h))

def is0(img, hsv, x, y):
    res = mask(img, hsv, [0,100,0], [179,255,255])
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret,binary_img=cv2.threshold(gray,32,255,cv2.THRESH_BINARY)

    #cv2.imwrite(str(x)+"_"+str(y)+".jpg", binary_img)

    if np.count_nonzero(binary_img)<10:
        return True
    
    return False

def get_kmeans_color(img, count, x, y):
    #t0 = time.time()
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters = count)
    clt.fit(img)

    hist = utils.centroid_histogram(clt)
    colors = clt.cluster_centers_.astype("uint8").tolist()

    if hist[0] > hist[1]:
        return [(hist[0], colors[0]),
                (hist[1], colors[1])]
    else:
        return [(hist[1], colors[1]),
                (hist[0], colors[0])]

    #print time.time() - t0

    #return zip(hist, clt.cluster_centers_.astype("uint8").tolist())

    #bar = utils.plot_colors(hist, clt.cluster_centers_)

    #cv2.imwrite(str(x)+"_"+str(y)+".jpg", bar)

def boximg2digit(main_colors, x, y):
    #print main_colors
    percent, color = main_colors[0]

    if color[1]<74:
        #print (x, y, color)
        return 0
    else:
        return -1

def process_mine_img(img, mine_area):
    digits = np.zeros((30,16), dtype=np.int8)*-1

    #get_kmeans_color(img,3)

    for y in range(0,16):
        for x in range(0,30):
            box_img = get_img_at_pos(img, mine_area, x, y)
            main_colors = get_kmeans_color(box_img, 2, x, y)
            digits[x][y] = boximg2digit(main_colors, x, y)
    return digits

#def onHMinChanged(x):
#    h=cv2.getTrackbarPos('H(Max)','image')
#    if x >= h:
#        cv2.setTrackbarPos('H(Min)','image',h-1)
#
#
#def onSMinChanged(x):
#    s=cv2.getTrackbarPos('S(Max)','image')
#    if x >= s:
#        cv2.setTrackbarPos('S(Min)','image',s-1)
#
#def onVMinChanged(x):
#    v=cv2.getTrackbarPos('V(Max)','image')
#    if x >= v:
#        cv2.setTrackbarPos('V(Min)','image',v-1)
#
#def onHMaxChanged(x):
#    h=cv2.getTrackbarPos('H(Min)','image')
#    if x <= h:
#        cv2.setTrackbarPos('H(Max)','image',h+1)
#
#def onSMaxChanged(x):
#    s=cv2.getTrackbarPos('S(Min)','image')
#    if x <= s:
#        cv2.setTrackbarPos('S(Max)','image',s+1)
#
#def onVMaxChanged(x):
#    v=cv2.getTrackbarPos('V(Min)','image')
#    if x <= v:
#        cv2.setTrackbarPos('V(Max)','image',v+1)

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

#cv2.namedWindow('image')
#
#cv2.createTrackbar('H(Min)','image',0,179,onHMinChanged)
#cv2.createTrackbar('H(Max)','image',179,179,onHMaxChanged)
#
#cv2.createTrackbar('S(Min)','image',100,255,onSMinChanged)
#cv2.createTrackbar('S(Max)','image',255,255,onSMaxChanged)
#
#cv2.createTrackbar('V(Min)','image',0,255,onVMinChanged) 
#cv2.createTrackbar('V(Max)','image',255,255,onVMaxChanged)

def draw_digit_at_pos(blank, digit, x, y):
    h0 = blank.shape[0]
    w0 = blank.shape[1]
    
    box_w = 1.0*w0/30
    box_h = 1.0*h0/16

    box_x = int(box_w*x)
    box_y = int(box_h*y)

    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank, str(digit),(box_x+8,box_y+24), font, 0.8,(0,0,255),2)

def do_test():
    if not set_dpi_aware():
        return

    hwnd = find_mine()
    l, t, w, h = get_client_rect(hwnd)

    handle_client_dc = win32gui.GetDC(hwnd)
    client_dc = win32ui.CreateDCFromHandle(handle_client_dc)
    mem_dc = client_dc.CreateCompatibleDC()

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(client_dc, w, h)
    mem_dc.SelectObject(bmp)

    mine_area = get_mine_area(get_shot(bmp, mem_dc, client_dc, l, t, w, h))
    area_x, area_y, area_w, area_h = mine_area

    #blank = cv2.imread("blank.jpg")
    #draw_digit_at_pos(blank, 2, 5, 5)

    while True:
        k=cv2.waitKey(1)&0xFF 
        if k==27: 
            break 
#
#        h_min=cv2.getTrackbarPos('H(Min)','image') 
#        s_min=cv2.getTrackbarPos('S(Min)','image') 
#        v_min=cv2.getTrackbarPos('V(Min)','image')
#        h_max=cv2.getTrackbarPos('H(Max)','image') 
#        s_max=cv2.getTrackbarPos('S(Max)','image') 
#        v_max=cv2.getTrackbarPos('V(Max)','image')
#        h_min=0
#        s_min=85
#        v_min=0
#        h_max=179
#        s_max=255
#        v_max=255

        # get window shot
        img_shot = get_shot(bmp, mem_dc, client_dc, l, t, w, h)
        blank = cv2.cvtColor(img_shot[area_y:area_y+area_h, area_x:area_x+area_w], cv2.COLOR_RGB2BGR)
        img_shot = cv2.cvtColor(img_shot, cv2.COLOR_RGB2HSV)
        #img_shot = img_shot[area_y:area_y+area_h,area_x:area_x+area_w]
        #rgb = cv2.cvtColor(img_shot, cv2.COLOR_RGB2BGR)
        #hsv=cv2.cvtColor(img_shot,cv2.COLOR_RGB2HSV)
        #res = mask(rgb,hsv,[h_min,s_min,v_min],[h_max,s_max,v_max])

        
        # 1 - 8  -> 1~8
        # 9 -> unkown

        # 10 -> known, 0
        # 11 -> known, mine

        # 0
        #((0,179),(100,255),(0,255))

        # 1
        #((0,179),(215,255),(0,255))

        # 2
        #((8,58),(215,255),(0,255))

        # 3 & mine
        #((0,8),(215,255),(0,255))

        # 4
        #((115,175),(215,255),(0,255))

        # 6
        #((58,115),(215,255),(0,255))

        # 9
        #((0,179),(0,90),(0,255))

        
        #cv2.resizeWindow('image',1920,800)
        # process
        digits = process_mine_img(img_shot, mine_area)
        for y in range(0,16):
            for x in range(0,30):
                if digits[x][y]==0:
                    draw_digit_at_pos(blank, 0, x, y)

        cv2.imshow('image',blank)


    mem_dc.DeleteDC()
    client_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, handle_client_dc)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    do_test()