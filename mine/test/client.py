import ctypes
import win32gui, win32ui, win32con, win32api

import time

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

def do_test():
    hwnd_mine = find_mine()
    set_dpi_aware()

    while(True):
        
        client_left, client_top, client_width, client_height = get_client_rect(hwnd_mine)
        print client_left, client_top, client_width, client_height

if __name__ == "__main__":
    do_test()