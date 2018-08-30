import cv2
import numpy as np
from PIL import Image
import zbarlight

cap=cv2.VideoCapture(0)

while(1):
    ret,frame=cap.read()
    
    if not ret:
        continue

    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    code_message = zbarlight.scan_codes('qrcode', image)
    if code_message != None:
        print code_message[0].decode()

    cv2.imshow('res',frame)

    k=cv2.waitKey(5)&0xFF
    if k==27:
        break

cv2.destroyAllWindows()