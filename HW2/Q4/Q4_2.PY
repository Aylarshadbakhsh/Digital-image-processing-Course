import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
img = cv2.imread('bone-scan.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
row , col = img.shape[:2]
def filter1(img,a,b):
 c=len(a)
 if c==3 and b=='median':
    kernel1_size = 3
    row, col = img.shape[:2]
    img3 = np.zeros((row + (kernel1_size - 1), col + (kernel1_size - 1)), dtype=np.uint8)
    img3[int(kernel1_size / 2) + 1:row + int(kernel1_size / 2) + 1,
    int(kernel1_size / 2) + 1:col + int(kernel1_size / 2) + 1] = img[:, :]

    img_con = np.zeros((row, col), dtype=np.int8)
    for i in range(0, row):
     for j in range(0, col):
      img_con[i][j] = np.sum(np.multiply(img3[i:i + kernel1_size, j:j + kernel1_size],a))
    img = cv2.medianBlur(img, 7)
    return img_con, img
 else:
         return False
img_con=np.zeros((row, col), dtype=np.uint8)
img1=np.zeros((row, col), dtype=np.uint8)
a=np.array([[0,-1,0],[-1,4,-1],[1,1,1]])
img_con,img1=filter1(img,a,'median')
cv2.imshow('conv',img_con)
cv2.imshow('median',img1)
cv2.imshow('original',img)
cv2.imwrite('conv.jpg',img_con)
cv2.imwrite('median.jpg',img1)
cv2.waitKey()