import numpy as np
import matplotlib.pyplot as plt
import cv2
img = cv2.imread('noisy_rectangle.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.title('original image')
plt.imshow(img,cmap='gray')
plt.axis(False)
plt.show()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(39,44))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(29,32))
#Erosion
erosion = cv2.erode(img,kernel,iterations = 1)
plt.imshow(erosion , cmap='gray')
plt.title('Erosion')
plt.axis(False)
plt.show()
#Dilation
dilation = cv2.dilate(img,kernel,iterations = 1)
plt.imshow(dilation , cmap='gray')
plt.axis(False)
plt.title('dilation')
plt.show()
#Opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
plt.imshow(opening , cmap='gray')
plt.axis(False)
plt.title('opening')
plt.show()
#Closing
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel3)
plt.imshow(closing , cmap='gray')
plt.axis(False)
plt.title('closing')
plt.show()


