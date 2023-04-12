import cv2
import numpy as np
import matplotlib.pyplot as plt
img1= cv2.imread('kidney.tif')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
row1, col1 = img1.shape[:2]

b1 = np.array(np.zeros((row1, col1), dtype=np.uint8))
b2 = np.array(np.zeros((row1, col1), dtype=np.uint8))

for i in range(0, row1):
    for j in range(0, col1):

        if (img1[i][j] >= 160 and img1[i][j] <= 240):
            b1[i][j] = 150
        else:
            b1[i][j] = 20

        if (img1[i][j] >= 100 and img1[i][j] <= 165):
            b2[i][j] = 200
        else:
            b2[i][j] = img1[i][j]
plt.figure(3)
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('original')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(b1, cmap='gray')
plt.title('left graph')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,3)
plt.imshow(b2,cmap='gray')
plt.title('right graph')
plt.xticks([])
plt.yticks([])
plt.show()