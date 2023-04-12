import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('MRIF.png')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('MRIS.png')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


plt.figure()
plt.subplot(121)
plt.imshow(img1)
plt.title('MRI1')
plt.axis(False)
plt.subplot(122)
plt.imshow(img2)
plt.title('MRI2')
plt.axis(False)
plt.show()

img_canny1 = cv2.Canny(gray1,100,100)
img_canny2 = cv2.Canny(gray2,100,100)

plt.figure()
plt.subplot(121)
plt.imshow(img_canny1 , cmap = 'gray')
plt.title('MRI1')
plt.axis(False)
plt.subplot(122)
plt.imshow(img_canny2 , cmap = 'gray')
plt.title('MRI2')
plt.axis(False)
plt.show()


points1 = np.float32([[65,79],[116,81],[104,135]])
points2 = np.float32([[341,192],[255,155],[218,231]])
matrix = cv2.getAffineTransform(points1,points2)

c_pts1 = np.float32([[65,79 , 1],[116,81,1],[104,135,1]])
c_pts1 = np.linalg.inv(c_pts1)

cte = np.zeros((2,3),dtype = np.float32)

for i in range(0,3):
    S1 = np.sum( c_pts1[i,:]*points2[:,0])
    S2 = np.sum( c_pts1[i,:]*points2[:,1])
    cte[0][i] = S1
    cte[1][i] = S2

test1 = cv2.warpAffine(gray1, matrix, (gray2.shape[1], gray2.shape[0]))
test2 = cv2.warpAffine(gray1, cte, (gray2.shape[1], gray2.shape[0]))


plt.figure()
plt.subplot(221)
plt.imshow(gray1 , cmap = 'gray')
plt.title('Original Image')
plt.axis(False)
plt.subplot(223)
plt.imshow(test1 , cmap = 'gray')
plt.title('Using Get Affine')
plt.axis(False)
plt.subplot(222)
plt.imshow(img2)
plt.title('MRI2')
plt.axis(False)
plt.subplot(224)
plt.imshow(test2 , cmap = 'gray')
plt.title('with Calculating the Matrix')
plt.axis(False)
plt.show()