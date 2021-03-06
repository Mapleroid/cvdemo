import time
import cv2
import numpy as np
from PIL import Image

import ctypes
import win32gui, win32ui, win32con, win32api

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

def boximg2digit(box_img, x, y):
    gray_img = cv2.cvtColor(box_img, cv2.COLOR_RGB2GRAY)
    edge_img = adaptiveThresholdimg2binary(gray_img)
    contours_img, contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(box_img,contours, -1,(0,0,255), 2)
    cv2.imwrite(str(x)+"_"+str(y)+".jpg", box_img)

    return 1

def process_mine_img(img, mine_area):
    digits = np.zeros((30,16), dtype=np.uint8)

    for y in range(0,16):
        for x in range(0,30):
            box_img = get_img_at_pos(img, mine_area, x, y)
            #cv2.imwrite(str(x)+"_"+str(y)+".jpg", box_img)
            digits[x][y] = boximg2digit(box_img, x, y)
    return digits

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

    while True:
        # get window shot
        img_shot = get_shot(bmp, mem_dc, client_dc, l, t, w, h)
        
        # process
        digits = process_mine_img(img_shot, mine_area)
        print digits
        break


    mem_dc.DeleteDC()
    client_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, handle_client_dc)

if __name__ == "__main__":
    do_test()