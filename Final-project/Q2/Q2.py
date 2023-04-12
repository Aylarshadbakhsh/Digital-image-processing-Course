#problem 2
import cv2
import numpy as np
import matplotlib.pyplot as plt
global img
def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
     print('Seed: ' + str(x) + ', ' + str(y), img[y, x], clicks.append((y, x)))
def dilt (img,seed):
   mask = np.zeros(img.shape[:2], np.uint8)
   mask[seed[0]:seed[0]+15,seed[1] :seed[1]+15]  = 255
   kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
   dilation= cv2.dilate(mask, kernel, iterations=8)
   #dilation = cv2.dilate(mask, kernel, iterations=30)
   image3=dilation+img
   return image3
image = cv2.imread('reflections.jpg', 0)
n=int(input("enter 0 for start(after every click close the window to see the result"))
while(n==0):
 clicks = []
 ret, img = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
 cv2.namedWindow('Input')
 cv2.setMouseCallback('Input', on_mouse, 0, )
 cv2.imshow('Input', img)
 k = cv2.waitKey(0)
 if k == 27:  # wait for ESC key to exit
     cv2.destroyAllWindows()
 seed = clicks[-1]
 out = dilt(img, seed)
 image=out


