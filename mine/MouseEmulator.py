import time
import win32gui, win32ui, win32con, win32api

LEFT_BUTTON = 0
RIGHT_BUTTON = 1
MIDDLE_BUTTON = 2

class MouseEmulator(object):

    def __init__(self, hwnd, mine_area):
        self._hwnd = hwnd
        self._mine_area = mine_area

    def mouse_click(self, button, point):
        # active window
        win32api.PostMessage(self._hwnd, win32con.WM_ACTIVATEAPP, 1, 0)

        # move cursor
        (origin_x, origin_y) = win32gui.GetCursorPos()
        screen_x, screen_y = win32gui.ClientToScreen(self._hwnd, point)
        win32api.SetCursorPos((screen_x, screen_y))

        # click
        if button==LEFT_BUTTON:
            down_msg = win32con.WM_LBUTTONDOWN
            up_msg = win32con.WM_LBUTTONUP
        elif button==RIGHT_BUTTON:
            down_msg = win32con.WM_RBUTTONDOWN
            up_msg = win32con.WM_RBUTTONUP
        else:
            down_msg = win32con.WM_MBUTTONDOWN
            up_msg = win32con.WM_MBUTTONUP

        win32api.PostMessage(self._hwnd, down_msg, 0, screen_x + screen_y*65536)
        time.sleep(0.05)
        win32api.PostMessage(self._hwnd, up_msg, 0, screen_x + screen_y*65536)

        # restore cursor
        win32api.SetCursorPos((origin_x, origin_y))

        # inactive window
        win32api.PostMessage(self._hwnd, win32con.WM_ACTIVATEAPP, 0, 0)

    def get_pos(self, mine_area, (col, row)):
        area_x, area_y, area_w, area_h = mine_area
        
        box_w = 1.0*area_w/30
        box_h = 1.0*area_h/16

        box_x = int(area_x + col*box_w)
        box_y = int(area_y + row*box_h)

        pos_x = box_x + box_w/2
        pos_y = box_y + box_h/2

        return (int(pos_x), int(pos_y))

    def click_cell(self, button, pos):
        point = self.get_pos(self._mine_area, pos)
        self.mouse_click(button, point)

    def left_click(self, cell_col, cell_row):
        self.click_cell(LEFT_BUTTON, (cell_col, cell_row))

    def right_click(self, cell_col, cell_row):
        self.click_cell(RIGHT_BUTTON, (cell_col, cell_row))

    def middle_click(self, cell_col, cell_row):
        self.click_cell(MIDDLE_BUTTON, (cell_col, cell_row))

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

    #cell_middle_click(hwnd_mine, mine_area, 2, 3)
    emulator = MouseEmulator(hwnd_mine, mine_area)

    for y in range(0,16):
        for x in range(0,30):
            emulator.right_click(x,y)
            time.sleep(0.2)
            #emulator.right_click(6,0)
            #time.sleep(1)
            #emulator.right_click(7,0)


    mem_dc.DeleteDC()
    client_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd_mine, handle_client_dc)
    
