import numpy as np
import cv2
from matplotlib import pyplot as plt

#WoxicDEV
#Instagram : @woxicdev
cap = cv2.VideoCapture(0)
while(True):
    
    ret, frame = cap.read()
    if frame is None:
        continue
    frame = cv2.resize(frame, (320,240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray.copy(),threshold1=50, threshold2=150,apertureSize = 3)
    laplacian = cv2.Laplacian(gray.copy(),cv2.CV_8U,13)
    im_th1 = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 5, 2)
    
    tempImg = cv2.medianBlur(gray, 5)
    im_th1 = cv2.adaptiveThreshold(tempImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 5, 2)
    blur = cv2.GaussianBlur(im_th1, (5, 5), 0)
    _, im_th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    numpy_horizontal = np.hstack((gray, edges, laplacian, im_th2))
    cv2.imshow('WoxicDEV', numpy_horizontal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()