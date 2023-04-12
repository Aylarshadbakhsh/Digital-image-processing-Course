import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('sonography.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
row , col = gray.shape


#canny
img_canny = cv2.Canny(img,100,100)

#sobel
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
img_sobel = np.absolute(sobelx) + np.absolute(sobely)

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
img_prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)
img_prewitt = np.absolute(img_prewittx) + np.absolute(img_prewitty)

#Roberts
roberts_v = np.array([[0,0,0],[0,1,0],[0,0,-1]])
roberts_h = np.array([[0,0,0],[0,0,1],[0,-1,0]])
vertical = cv2.filter2D(gray, cv2.CV_64F, roberts_v)
horizontal = cv2.filter2D(gray, cv2.CV_64F, roberts_h)
image_roberts = np.absolute(horizontal) + np.absolute(vertical)

#LOG
gaussian = cv2.GaussianBlur(gray,(3,3),0)
laplacian = cv2.Laplacian(gaussian,cv2.CV_64F,ksize=3)
ther = 0.35*np.max(laplacian)
output = np.zeros(laplacian.shape,np.uint8)

for i in range(1, row - 1):
    for j in range(1, col - 1):
        patch = laplacian[i-1:i+2, j-1:j+2]
        p = laplacian[i, j]
        maxP = patch.max()
        minP = patch.min()
        if (p > 0):
            zeroCross = True if minP < 0 else False
        else:
            zeroCross = True if maxP > 0 else False
        if ((maxP - minP) > ther) and zeroCross:
            output[i, j] = 1

_, sobThresh = cv2.threshold(img_sobel, 78, 255, cv2.THRESH_BINARY)
_, prewThresh = cv2.threshold(img_prewitt, 86, 255, cv2.THRESH_BINARY)
_, robertThresh = cv2.threshold(image_roberts, 25, 255, cv2.THRESH_BINARY)


plt.figure()
plt.subplot(231)
plt.imshow(gray , cmap='gray')
plt.title('Original')
plt.axis(False)
plt.subplot(232)
plt.imshow(img_canny , cmap='gray')
plt.title('Canny')
plt.axis(False)
plt.subplot(233)
plt.imshow(sobThresh, cmap='gray')
plt.title('Sobel')
plt.axis(False)
plt.subplot(234)
plt.imshow(prewThresh, cmap='gray')
plt.title('Prewitt')
plt.axis(False)
plt.subplot(235)
plt.imshow(robertThresh, cmap='gray')
plt.title('Roberts')
plt.axis(False)
plt.subplot(236)
plt.imshow(output, cmap='gray')
plt.title('LOG')
plt.axis(False)
plt.show()




