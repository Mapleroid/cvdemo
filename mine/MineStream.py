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
    #bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb

def get_img_at_pos(img_shot, mine_area, x, y):
    area_x, area_y, area_w, area_h = mine_area
    
    box_w = 1.0*area_w/30
    box_h = 1.0*area_h/16

    box_x = int(area_x + x*box_w)
    box_y = int(area_y + y*box_h)

    return img_shot[box_y:(box_y+int(box_h)), box_x:(box_x+int(box_w))]

def boximg2digit(main_colors, x, y, _logger):

    return -1

def get_kmeans_color2(img, count, x, y, _logger):
    #img, pos, count = args
    #x, y = pos
    
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    #t0 = time.time()
    clt = KMeans(n_clusters = count)
    clt.fit(img)
    #_logger.info("elapsed:%f", time.time()-t0)

    hist = utils.centroid_histogram(clt)
    colors = clt.cluster_centers_.astype("uint8").tolist()

    if hist[0] > hist[1]:
        return ((x,y), [(hist[0], colors[0]),(hist[1], colors[1])])
    else:
        return ((x,y), [(hist[1], colors[1]),(hist[0], colors[0])])

def get_kmeans_color(img, count):
    img = img.reshape((-1,3))
    img = np.float32(img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 20000.0)
    ret,label,center=cv2.kmeans(img, count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    count_num = np.bincount(label.flatten())
    count_num = count_num.astype("float")
    hist = count_num/count_num.sum()

    sorted_index = np.argsort(-hist)

    result = []
    for index in sorted_index:
        result.append( (center[index], hist[index]) )

    return result

def process_mine_img(img, mine_area, _logger):
    #t0 = time.time()
    digits = np.zeros((30,16), dtype=np.int8)*-1

#    box_imgs_args = []
#
#    for y in range(0,16):
#        for x in range(0,30):
#            box_img = get_img_at_pos(img, mine_area, x, y)
#            box_imgs_args.append( (box_img, (x,y), 2) )
    
    #with Pool(processes=8) as _pool:
    #_pool = Pool(processes=8)
    #print _pool.map(get_kmeans_color, box_imgs_args, 3)

    #pool = eventlet.GreenPool(1000)
    #for pos, main_colors in pool.imap(get_kmeans_color, box_imgs, kmeans_counts, positions, _loggers):
    #    print (pos, main_colors)

    for y in range(0,16):
        for x in range(0,30):
            box_img = get_img_at_pos(img, mine_area, x, y)
            #t = KMeansThread.KMeansThread(get_kmeans_color, (box_img, 2, x, y, _logger), str(x)+"_"+str(y))
            #process_threads.append(t)
            main_colors = get_kmeans_color(box_img, 3)
            #bar = utils.plot_colors2(main_colors)
            #cv2.imwrite(str(x)+"_"+str(y)+".jpg", bar)

            #print main_colors
            digits[x][y] = boximg2digit(main_colors, x, y, _logger)

    #_logger.info("elapsed:%f", time.time()-t0)
    return digits

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
        k=cv2.waitKey(1)&0xFF 
        if k==27: 
            break

        #t0 = time.time()
        
        img_shot = get_shot(bmp, mem_dc, client_dc, client_rect_left, client_rect_top, client_rect_width, client_rect_height)
        blank = cv2.cvtColor(img_shot[area_y:area_y+area_h, area_x:area_x+area_w], cv2.COLOR_RGB2BGR)
        img_shot = cv2.cvtColor(img_shot, cv2.COLOR_RGB2HSV)
        digits = process_mine_img(img_shot, mine_area, _logger)
        for y in range(0,16):
            for x in range(0,30):
                if digits[x][y]==0:
                    draw_digit_at_pos(blank, 0, x, y)

        cv2.imshow('image',blank)

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
