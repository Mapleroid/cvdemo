import time
import win32gui, win32ui, win32con, win32api
import ctypes

def find_mine():
    wndtitle = None
    wndclass = "Minesweeper"
    wnd = win32gui.FindWindow(wndclass, wndtitle)
    return wnd

def window_capture(hwnd):
    try:
        #f = ctypes.windll.dwmapi.DwmGetWindowAttribute
        f = ctypes.windll.shcore.SetProcessDpiAwareness
    except WindowsError:
        f = None

    if not f:
        exit(0)
    else:
        PROCESS_SYSTEM_DPI_AWARE = 1
        f(PROCESS_SYSTEM_DPI_AWARE)

    l, t, r, b = win32gui.GetClientRect(hwnd)
    w = r -l
    h = b-t

    hwnd_client_dc = win32gui.GetDC(hwnd)
    img_dc = win32ui.CreateDCFromHandle(hwnd_client_dc)
    mem_dc = img_dc.CreateCompatibleDC()

    screenshot = win32ui.CreateBitmap()
    screenshot.CreateCompatibleBitmap(img_dc, w, h)
    mem_dc.SelectObject(screenshot)

    mem_dc.BitBlt((0, 0), (w, h), img_dc, (l, t), win32con.SRCCOPY)

    screenshot.SaveBitmapFile(mem_dc, 'screenshot.bmp')
    mem_dc.DeleteDC()
    win32gui.DeleteObject(screenshot.GetHandle())


#    desktop_dc = win32gui.GetWindowDC(hdesktop)
#    img_dc = win32ui.CreateDCFromHandle(desktop_dc)
#    mem_dc = img_dc.CreateCompatibleDC()
#
#    screenshot = win32ui.CreateBitmap()
#    screenshot.CreateCompatibleBitmap(img_dc, d_width, d_height)
#    mem_dc.SelectObject(screenshot)
#
#    mem_dc.BitBlt((0, 0), (d_width, d_height), img_dc, (d_left, d_top), win32con.SRCCOPY)
#
#    screenshot.SaveBitmapFile(mem_dc, 'screenshot.bmp')
#    mem_dc.DeleteDC()
#    win32gui.DeleteObject(screenshot.GetHandle())


#    hwnd_dc = win32gui.GetWindowDC(hwnd)
#    img_dc = win32ui.CreateDCFromHandle(hwnd_dc)
#    mem_dc = img_dc.CreateCompatibleDC()
#
#    screenshot = win32ui.CreateBitmap()
#    screenshot.CreateCompatibleBitmap(img_dc, width, height)
#    mem_dc.SelectObject(screenshot)
#    mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)
#    screenshot.SaveBitmapFile(img_dc, 'screenshot.bmp')
#    mem_dc.DeleteDC()
#    win32gui.DeleteObject(screenshot.GetHandle())
#    hwnd_img = win32ui.CreateBitmap()
#    hwnd_img.CreateCompatibleBitmap(img_dc, w, h)
#
#    win32gui.SelectObject(img_dc, hwnd_img)
#    win32gui.BitBlt(img_dc, 0, 0, w, h, hwnd_dc, 0, 0, win32con.SRCCOPY)
#    hwnd_img.SaveBitmapFile(img_dc, "haha.jpg")

#    hwnd_dc = win32gui.GetWindowDC(hwnd)
#    img_dc = win32ui.CreateDCFromHandle(hwnd_dc)
#    mem_dc = img_dc.CreateCompatibleDC()
#
#    # bitmap
#    screenshot = win32ui.CreateBitmap()
#    screenshot.CreateCompatibleBitmap(img_dc, w, h)
#    mem_dc.SelectObject(screenshot)
#
#    # screenshot
#    #mem_dc.BitBlt((left, top), (w, h), img_dc, (0, 0), win32con.SRCCOPY)
#    win32gui.BitBlt(mem_dc,0,0,w,h,img_dc,-7,-7,win32con.SRCCOPY)
#    screenshot.SaveBitmapFile(mem_dc, "haha.jpg")


def test():
    beg = time.time()
    for i in range(10):
      window_capture("haha.jpg")
    end = time.time()
    print(end - beg)



if __name__ == "__main__":
    window_capture(find_mine())