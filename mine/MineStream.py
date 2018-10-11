import cv2
import time
import logging
import logging.handlers
import ctypes
import win32gui, win32ui, win32con, win32api

import numpy as np
from PIL import Image

import utils
import DispatchQueue
import Processor
import Params
import MouseEmulator

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
    return (l, t, r-l, b-t)

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
    mem_dc.BitBlt((0, 0), (w, h), client_dc, (l, t), win32con.SRCCOPY)
    bmpstr = bmp.GetBitmapBits(True)
    pil_img = Image.frombuffer('RGB',(w, h),bmpstr, 'raw', 'BGRX', 0, 1)
    return np.array(pil_img)


def draw_digit_at_pos(blank, digit, x, y):
    h0 = blank.shape[0]
    w0 = blank.shape[1]
    
    box_w = 1.0*w0/30
    box_h = 1.0*h0/16

    box_x = int(box_w*x)
    box_y = int(box_h*y)

    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank, str(digit),(box_x+8,box_y+24), font, 0.8,(0,0,255),2)

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

    client_left, client_top, client_width, client_height = get_client_rect(hwnd_mine)

    handle_client_dc = win32gui.GetDC(hwnd_mine)
    client_dc = win32ui.CreateDCFromHandle(handle_client_dc)
    mem_dc = client_dc.CreateCompatibleDC()

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(client_dc, client_width, client_height)
    mem_dc.SelectObject(bmp)

    mine_area=None
    while(True):
        img_shot = get_shot(bmp, mem_dc, client_dc, client_left, client_top, client_width, client_height)
        mine_area = get_mine_area(img_shot)
        if mine_area is not None:
            break

    area_x, area_y, area_w, area_h = mine_area
    emulator = MouseEmulator.MouseEmulator(hwnd_mine, mine_area)
    #_logger.info("work area width:%d, height:%d", area_w, area_h)

    # main loop

    split_flag = False
    test_flag =  False
    convert_flag = False 
    solve_flag = True

    if test_flag:
        Params.create_params_window()

    while(True):
        k=cv2.waitKey(1)&0xFF 
        if k==27: 
            break

        #t0 = time.time()
        img_shot = get_shot(bmp, mem_dc, client_dc, client_left, client_top, client_width, client_height)[area_y:area_y+area_h, area_x:area_x+area_w]

        if split_flag:
            split_flag = False
            box_imgs = Processor.split_mine_img(img_shot)
            for (x,y), box_img in box_imgs:
                cv2.imwrite(str(x)+"_"+str(y)+".jpg", box_img)

        if test_flag:
            params =  Params.get_params()
            img_shot = Processor.process_mine_img(img_shot, params)
            cv2.imshow('res',img_shot)

        if convert_flag:
            blank = cv2.cvtColor(img_shot, cv2.COLOR_RGB2BGR)
            mines, digits = Processor.img2digits(img_shot)
            for y in range(0,16):
                for x in range(0,30):
                    if digits[x][y]>0:
                        draw_digit_at_pos(blank, digits[x][y], x, y)
                    elif digits[x][y] == -2:
                        draw_digit_at_pos(blank, 0, x, y)


            cv2.imshow('res',blank)

        if solve_flag:
            #solve_flag = False
            Processor.solve_game_one_step(img_shot, emulator)
            time.sleep(0.2)
            #solve_flag = True
            #time.sleep(0.5)
        #hsv = cv2.cvtColor(img_shot, cv2.COLOR_RGB2HSV)
        #res = Processor.mask(hsv, hsv, [h_min, s_min, v_min],[h_max, s_max, v_max])
        #res = Processor.mask(img_shot, hsv, [0, 0, 175],[179, 35, 255])
        #gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY) 
        #ret, binary = cv2.threshold(gray,32,255,cv2.THRESH_BINARY)
        #binary = 255-binary
        #img = res[area_y:area_y+area_h, area_x:area_x+area_w]
        #blank = cv2.cvtColor(img_shot, cv2.COLOR_RGB2BGR)

#        digits = Processor.process_mine_img(img_shot, params)
#        for y in range(0,16):
#            for x in range(0,30):
#                if digits[x][y]==0:
#                    draw_digit_at_pos(blank, 0, x, y)
#                if digits[x][y]==9:
#                    draw_digit_at_pos(blank, 9, x, y)

        #cv2.imshow('image',blank)
        #Params.show_params_window()
        
#        img_shot = Processor.split_mine_img(img_shot)
#        if not flag:
#            flag = True
#            cv2.imwrite("test.jpg", img_shot)
        
        #gray = cv2.cvtColor(img_shot, cv2.COLOR_RGB2GRAY)
        #cv2.imshow('res',img_shot)
        #_logger.info("elapsed:%f", time.time()-t0)

        
    mem_dc.DeleteDC()
    client_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd_mine, handle_client_dc)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logger = logging.getLogger("minesweaper_log")
    logger.setLevel(level = logging.DEBUG)

    # set file log level to info
    file_handler = logging.handlers.RotatingFileHandler('mine.log', maxBytes=10*1024*1024,backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_handler_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_handler_formatter)
    logger.addHandler(file_handler)

    # set console log level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler_formatter = logging.Formatter('%(asctime)s - %(filename)s, %(lineno)d - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_handler_formatter)
    logger.addHandler(console_handler)

    start_work()
