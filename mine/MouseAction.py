import time
import ctypes
import win32gui, win32ui, win32con, win32api

LEFT_BUTTON = 0
RIGHT_BUTTON = 1
MIDDLE_BUTTON = 2

def mouse_left_click(hwnd, pos_x=None,pos_y=None):
    if pos_x is not None and pos_y is not None:
        win32api.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, 0, pos_x + pos_y*65536)
        time.sleep(0.05)
        win32api.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, pos_x + pos_y*65536)

def mouse_right_click(hwnd, pos_x=None,pos_y=None):
    if pos_x is not None and pos_y is not None:
        win32api.PostMessage(hwnd, win32con.WM_RBUTTONDOWN, 0, pos_x + pos_y*65536)
        time.sleep(0.1)
        win32api.PostMessage(hwnd, win32con.WM_RBUTTONUP, 0, pos_x + pos_y*65536)

def mouse_middle_click(hwnd, pos_x=None,pos_y=None):
    if pos_x is not None and pos_y is not None:
        win32api.PostMessage(hwnd, win32con.WM_MBUTTONDOWN, 0, pos_x + pos_y*65536)
        time.sleep(0.05)
        win32api.PostMessage(hwnd, win32con.WM_MBUTTONUP, 0, pos_x + pos_y*65536)

def mouse_click(hwnd, point, button):
    # active window
    win32api.PostMessage(hwnd, win32con.WM_ACTIVATEAPP, 1, 0)

    # move cursor
    (origin_x, origin_y) = win32gui.GetCursorPos()
    screen_x, screen_y = win32gui.ClientToScreen(hwnd, point)
    win32api.SetCursorPos((screen_x, screen_y))

    # click
    if button==LEFT_BUTTON:
        mouse_left_click(hwnd, screen_x, screen_y)
    elif button==RIGHT_BUTTON:
        mouse_right_click(hwnd, screen_x, screen_y)
    else:
        mouse_middle_click(hwnd, screen_x, screen_y)

    # restore cursor
    win32api.SetCursorPos((origin_x, origin_y))

    # inactive window
    win32api.PostMessage(hwnd, win32con.WM_ACTIVATEAPP, 0, 0)

def get_pos(mine_area, x, y):
    area_x, area_y, area_w, area_h = mine_area
    
    box_w = 1.0*area_w/30
    box_h = 1.0*area_h/16

    box_x = int(area_x + x*box_w)
    box_y = int(area_y + y*box_h)

    pos_x = box_x + box_w/2
    pos_y = box_y + box_h/2

    return int(pos_x), int(pos_y)

def cell_left_click(hwnd, mine_area, cell_col, cell_row):
    pos_x, pos_y = get_pos(mine_area, cell_col, cell_row)
    mouse_click(hwnd, (pos_x, pos_y), LEFT_BUTTON)

def cell_right_click(hwnd, mine_area, cell_col, cell_row):
    pos_x, pos_y = get_pos(mine_area, cell_col, cell_row)
    mouse_click(hwnd, (pos_x, pos_y), RIGHT_BUTTON)

def cell_middle_click(hwnd, mine_area, cell_col, cell_row):
    pos_x, pos_y = get_pos(mine_area, cell_col, cell_row)
    mouse_click(hwnd, (pos_x, pos_y), MIDDLE_BUTTON)

if __name__ == "__main__":
    import MineStream

    hwnd_mine = MineStream.find_mine()
    if hwnd_mine ==0:
        print "error get hwnd"
        exit(0)

    if not MineStream.set_dpi_aware():
        print "error set dpi awareness"
        exit(0)

    client_rect_left, client_rect_top, client_rect_width, client_rect_height = MineStream.get_client_rect(hwnd_mine)

    handle_client_dc = win32gui.GetDC(hwnd_mine)
    client_dc = win32ui.CreateDCFromHandle(handle_client_dc)
    mem_dc = client_dc.CreateCompatibleDC()

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(client_dc, client_rect_width, client_rect_height)
    mem_dc.SelectObject(bmp)

    mine_area=None
    while(True):
        img_shot = MineStream.get_shot(bmp, mem_dc, client_dc, client_rect_left, client_rect_top, client_rect_width, client_rect_height)
        mine_area = MineStream.get_mine_area(img_shot)
        if mine_area is not None:
            break

    #print mine_area

    
    #right_click(hwnd_mine, mine_area, 0, 1)

    cell_middle_click(hwnd_mine, mine_area, 2, 3)


    mem_dc.DeleteDC()
    client_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd_mine, handle_client_dc)
    
