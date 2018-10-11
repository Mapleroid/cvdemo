import cv2
import numpy as np
import time

def mask(bgr_img,hsv_img,low,up):
    lower_color = np.array(low)
    upper_color = np.array(up)
    mask_color = cv2.inRange(hsv_img,lower_color,upper_color)
    return cv2.bitwise_and(bgr_img,bgr_img,mask=mask_color)

def split_mine_img(img):
    outer_border = 4.5
    inner_border = 1.5

    h0 = img.shape[0] - outer_border*2
    w0 = img.shape[1] - outer_border*2
    average_w = 1.0*w0/30
    average_h = 1.0*h0/16

    box_w = int(average_w - inner_border*2)
    box_h = int(average_h - inner_border*2)

    box_imgs = []
    for y in range(0,16):
        for x in range(0,30):
            box_x = int(x*average_w+inner_border+outer_border)
            box_y = int(y*average_h+inner_border+outer_border)
            box_imgs.append( ((x,y) ,img[box_y:box_y+box_h, box_x:box_x+box_w]) )

    return box_imgs

def isOpened(box_img, hsv_box_img):
    masked_box_img = mask(box_img, hsv_box_img, [100,0,0],[120,60,255])
    gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY)

    flat_img = binary_box_img.flatten()
    return 1.0*np.count_nonzero(flat_img)/flat_img.shape[0]>0.2

def isOne(box_img, hsv_box_img):
    masked_box_img = mask(box_img, hsv_box_img, [0,110,180],[179,255,255])
    gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY)

    flat_img = binary_box_img.flatten()
    return 1.0*np.count_nonzero(flat_img)/flat_img.shape[0]>0.1

def isTwo(box_img, hsv_box_img):
    masked_box_img = mask(box_img, hsv_box_img, [50,20,0],[70,255,255])
    gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY)

    flat_img = binary_box_img.flatten()
    return 1.0*np.count_nonzero(flat_img)/flat_img.shape[0]>0.1

def isThree(box_img, hsv_box_img):
    masked_box_img = mask(box_img, hsv_box_img, [0,225,160],[179,255,255])
    gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY)

    flat_img = binary_box_img.flatten()
    return 1.0*np.count_nonzero(flat_img)/flat_img.shape[0]>0.1

def isFour(box_img, hsv_box_img):
    #masked_box_img = mask(box_img, hsv_box_img, [115,60,0],[125,255,175])
    #gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    #ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY)
    masked_box_img = mask(box_img, hsv_box_img, [0,0,0],[179,255,255])
    gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY_INV)

    flat_img = binary_box_img.flatten()
    return 1.0*np.count_nonzero(flat_img)/flat_img.shape[0]>0.1

def isFive(box_img, hsv_box_img):
    masked_box_img = mask(box_img, hsv_box_img, [0,200,0],[40,255,135])
    gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY)

    flat_img = binary_box_img.flatten()
    return 1.0*np.count_nonzero(flat_img)/flat_img.shape[0]>0.1

def isSix(box_img, hsv_box_img):
    masked_box_img = mask(box_img, hsv_box_img, [70,100,0],[100,255,255])
    gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY)

    flat_img = binary_box_img.flatten()
    return 1.0*np.count_nonzero(flat_img)/flat_img.shape[0]>0.1

def isFlag(box_img, hsv_box_img):
    masked_box_img = mask(box_img, hsv_box_img, [0,0,0],[85,255,255]) + mask(box_img, hsv_box_img, [135,0,0],[179,255,255])
    gray_box_img = cv2.cvtColor(masked_box_img, cv2.COLOR_RGB2GRAY)
    ret, binary_box_img = cv2.threshold(gray_box_img,32,255,cv2.THRESH_BINARY)

    flat_img = binary_box_img.flatten()
    return 1.0*np.count_nonzero(flat_img)/flat_img.shape[0]>0.1

def box2digit(box_img):
    # (mine, digit)
    # mine = 0 , not mine
    # mine = 1, mine
    hsv_box_img = cv2.cvtColor(box_img, cv2.COLOR_RGB2HSV)

    if isOpened(box_img, hsv_box_img):
        if isOne(box_img, hsv_box_img):
            return (0, 1)

        if isTwo(box_img, hsv_box_img):
            return (0, 2)

        if isThree(box_img, hsv_box_img):
            return (0, 3)

        if isFour(box_img, hsv_box_img):
            return (0, 4)

        if isFive(box_img, hsv_box_img):
            return (0, 5)

        if isSix(box_img, hsv_box_img):
            return (0, 6)
        else:
            return (0, 0)
    else:
        if isFlag(box_img, hsv_box_img):
            return (1, -1)

        return (0, -1)

def img2digits(img):
    mines = np.zeros((30,16), dtype=np.int8)
    digits = np.ones((30,16), dtype=np.int8)*-1
    box_imgs = split_mine_img(img)

    for (x,y), box_img in box_imgs:
        mines[x][y], digits[x][y] = box2digit(box_img)

    return mines, digits


def known_mines_around(mines, x0, y0):
    x1 = x0-1
    y1 = y0-1
    x2 = x0+1
    y2 = y0+1

    if x1<0:
        x1=0
    
    if y1<0:
        y1=0

    if x2>29:
        x2=29

    if y2>15:
        y2=15

    count = 0
    for x in range(x1, x2+1):
        for y in range(y1, y2+1):
            if x==x0 and y==y0:
                continue

            count += mines[x][y]

    return count

def unknown_mines(mines, digits):
    tmp = mines+digits
    for y in range(0,16):
        for x in range(0,30):
            if tmp[x][y]==-1:
                tmp[x][y]=1
            else:
                tmp[x][y]=0

    return tmp

def label_flags(emulator, unknows, is_labeled, x0, y0):
    x1 = x0-1
    y1 = y0-1
    x2 = x0+1
    y2 = y0+1

    if x1<0:
        x1=0
    
    if y1<0:
        y1=0

    if x2>29:
        x2=29

    if y2>15:
        y2=15

    for x in range(x1, x2+1):
        for y in range(y1, y2+1):
            #print (x,y)
            
            if unknows[x][y] == 1:
                if is_labeled[x][y]==0:
                    emulator.right_click(x,y)
                    is_labeled[x][y]=1
            #    time.sleep(3)

def solve_game_one_step(img, emulator):
    mines, digits = img2digits(img)
    unknows = unknown_mines(mines, digits)

    aroud_mines = np.ones((30,16), dtype=np.int8)*-1
    aroud_unknows = np.ones((30,16), dtype=np.int8)*-1
    is_labeled = np.zeros((30,16), dtype=np.int8)

    for y in range(0,16):
        for x in range(0,30):
            aroud_mines[x][y] = known_mines_around(mines, x, y)
            aroud_unknows[x][y] = known_mines_around(unknows, x, y)

            if aroud_unknows[x][y]>0 and aroud_mines[x][y] == digits[x][y]:
                emulator.middle_click(x,y)
                time.sleep(0.1)

    #for y in range(0,16):
    #    for x in range(0,30):
            if digits[x][y]>0 and aroud_unknows[x][y]>0 and aroud_unknows[x][y]==digits[x][y]-aroud_mines[x][y]:
                #print (x,y)
                #print (digits[x][y], aroud_unknows[x][y], aroud_mines[x][y])
                label_flags(emulator, unknows, is_labeled, x, y)
            #if digits[x][y]>0 and aroud_mines[x][y] == digits[x][y]:
            #    emulator.middle_click(x,y)
            #    return

    #print (digits[3][1], aroud_unknows[3][1], aroud_mines[3][1])
    #diffs = digits - aroud_mines
    #for y in range(0,16):
    #    for x in range(0,30):
    

#    if digits[x][y]>0 and aroud_mines[x][y] == mines[x][y]:
#                emulator.middle_click(x,y)
#                return
    #print mines.T
    #print digits.T
    #print aroud_unknows
    #print diffs

def process_mine_img(img, params):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    (h_min, h_max), (s_min, s_max), (v_min, v_max), (t_min, t_max) = params
    mask_img = mask(img, hsv, [h_min, s_min, v_min],[h_max, s_max, v_max])
    #mask_img = mask(img, hsv, [100,0,0],[120,60,255])
    #mask_hsv = mask(hsv, hsv, [h_min, s_min, v_min],[h_max, s_max, v_max])

    gray_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray_img,32,255,cv2.THRESH_BINARY)

    #binary_box_imgs = split_mine_img(binary)
    #hsv_box_imgs = split_mine_img(mask_hsv)

    return binary

def process_test(img):
    return img