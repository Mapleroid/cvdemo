import cv2
import numpy as np

def onHMinChanged(x):
    h=cv2.getTrackbarPos('H(Max)','Params')
    if x >= h:
        cv2.setTrackbarPos('H(Min)','Params',h-1)

def onSMinChanged(x):
    s=cv2.getTrackbarPos('S(Max)','Params')
    if x >= s:
        cv2.setTrackbarPos('S(Min)','Params',s-1)

def onVMinChanged(x):
    v=cv2.getTrackbarPos('V(Max)','Params')
    if x >= v:
        cv2.setTrackbarPos('V(Min)','Params',v-1)

def onHMaxChanged(x):
    h=cv2.getTrackbarPos('H(Min)','Params')
    if x <= h:
        cv2.setTrackbarPos('H(Max)','Params',h+1)

def onSMaxChanged(x):
    s=cv2.getTrackbarPos('S(Min)','Params')
    if x <= s:
        cv2.setTrackbarPos('S(Max)','Params',s+1)

def onVMaxChanged(x):
    v=cv2.getTrackbarPos('V(Min)','Params')
    if x <= v:
        cv2.setTrackbarPos('V(Max)','Params',v+1)

def onTMinChanged(x):
    t=cv2.getTrackbarPos('T(Max)','Params')
    if x >= t:
        cv2.setTrackbarPos('T(Min)','Params',t-1)

def onTMaxChanged(x):
    t=cv2.getTrackbarPos('T(Min)','Params')
    if x <= t:
        cv2.setTrackbarPos('T(Max)','Params',t+1)

def create_params_window():
    cv2.namedWindow('Params')

    cv2.createTrackbar('H(Min)','Params',0,179,onHMinChanged)
    cv2.createTrackbar('H(Max)','Params',179,179,onHMaxChanged)

    cv2.createTrackbar('S(Min)','Params',0,255,onSMinChanged)
    cv2.createTrackbar('S(Max)','Params',255,255,onSMaxChanged)

    cv2.createTrackbar('V(Min)','Params',0,255,onVMinChanged) 
    cv2.createTrackbar('V(Max)','Params',255,255,onVMaxChanged)

    cv2.createTrackbar('T(min)','Params',100,1000,onTMinChanged)
    cv2.createTrackbar('T(max)','Params',200,1000,onTMaxChanged)

def get_params():
    h_min=cv2.getTrackbarPos('H(Min)','Params') 
    s_min=cv2.getTrackbarPos('S(Min)','Params') 
    v_min=cv2.getTrackbarPos('V(Min)','Params')
    h_max=cv2.getTrackbarPos('H(Max)','Params') 
    s_max=cv2.getTrackbarPos('S(Max)','Params') 
    v_max=cv2.getTrackbarPos('V(Max)','Params')
    t_min=cv2.getTrackbarPos('T(min)','Params')
    t_max=cv2.getTrackbarPos('T(max)','Params')

    return ( (h_min, h_max), (s_min, s_max), (v_min, v_max), (t_min, t_max) )

def show_params_window():
    bar = np.zeros((50, 300, 3), dtype = "uint8")

    cv2.imshow('Params', bar)
    #cv2.resizeWindow('Params',1920,960)

def test():
    create_params_window()

    while True:
        k=cv2.waitKey(1)&0xFF 
        if k==27: 
            break

        (h_min, h_max), (s_min, s_max), (v_min, v_max), (t_min, t_max) = get_params()
        show_params_window()
        

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()