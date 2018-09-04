import cv2
import time
import logging
import logging.handlers
import ctypes
import win32gui, win32ui, win32con, win32api

import numpy as np
from PIL import Image

import DispatchQueue

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

def get_mine_area(bgr_shot):
    gray_img = cv2.cvtColor(bgr_shot, cv2.COLOR_BGR2GRAY)
    edge_img = adaptiveThresholdimg2binary(gray_img)
    contours_img, contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_idxs = []
    for index in xrange(len(contours)):
        arcl = cv2.arcLength(contours[index],True)
        if arcl > (bgr_shot.shape[0]+bgr_shot.shape[1])*2*0.8:
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
    bmpstr = bmp.GetBitmapBits(True)
    pil_img = Image.frombuffer('RGB',(w, h),bmpstr, 'raw', 'BGRX', 0, 1)
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def start_work():
    _logger = logging.getLogger("minesweaper_log")
    _logger.info("stream reader starting...")

    hwnd_mine = find_mine()
    if hwnd_mine==0:
        _logger.info("can not find minesweaper game.")

        while(True):
            time.sleep(1)

    _logger.info("find game %d", hwnd_mine)
    if not set_dpi_aware():
        _logger.info("can not set dpi awareness.")
        
        while(True):
            time.sleep(1)

    client_rect_left, client_rect_top, client_rect_width, client_rect_height = get_client_rect(hwnd_mine)

    handle_client_dc = win32gui.GetDC(hwnd_mine)
    client_dc = win32ui.CreateDCFromHandle(handle_client_dc)
    mem_dc = client_dc.CreateCompatibleDC()

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(client_dc, client_rect_width, client_rect_height)
    mem_dc.SelectObject(bmp)

    mine_area=None
    while(True):
        img_shot = get_shot(bmp, mem_dc, client_dc, client_rect_left, client_rect_top, client_rect_width, client_rect_height)
        mine_area = get_mine_area(img_shot)
        if mine_area is not None:
            break

    area_x, area_y, area_w, area_h = mine_area
    _logger.info("work area width:%d, height:%d", area_w, area_h)


    # main loop
    while(True):
#        k=cv2.waitKey(1)&0xFF 
#        if k==27: 
#            break

        #t0 = time.time()
        img_shot = get_shot(bmp, mem_dc, client_dc, client_rect_left, client_rect_top, client_rect_width, client_rect_height)
        #cv2.rectangle(img_shot, (area_x, area_y), (area_x+area_w, area_y+area_h), (0,0,255), 3)
        #cv2.imshow('image',img_shot)
        #_logger.info("elapsed:%f", time.time()-t0)


    mem_dc.DeleteDC()
    client_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd_mine, handle_client_dc)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    logger = logging.getLogger("minesweaper_log")
    logger.setLevel(level = logging.DEBUG)

    # set file log level to info
    file_handler = logging.handlers.RotatingFileHandler('mine.log', maxBytes=10*1024*1024,backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_handler_formatter = logging.Formatter('%(asctime)s - %(filename)s, %(lineno)d - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_handler_formatter)
    logger.addHandler(file_handler)

    # set console log level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler_formatter = logging.Formatter('%(asctime)s - %(filename)s, %(lineno)d - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_handler_formatter)
    logger.addHandler(console_handler)



    start_work()
