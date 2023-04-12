import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('lung.png')
img1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_2 = cv2.blur(img, (3, 3))
median=cv2.medianBlur(img,3)
img_3 = cv2.Sobel(median ,cv2.CV_64F, dx=1, dy=0)#both
img_3=np.uint8(np.absolute(img_3))
img_4=cv2.Sobel(median,cv2.CV_8U,dx=1,dy=0)#positive slope
plt.figure()
plt.suptitle('Problem 3 Figure')

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.axis(False)

plt.subplot(1, 3, 2)
plt.title('Denoised')
plt.imshow(img_2, cmap='gray')
plt.axis(False)

plt.subplot(1, 3, 3)
plt.title('Gradient')
plt.imshow(img_3, cmap='gray')
plt.axis(False)

plt.show()
plt.subplot(1, 3, 1)
plt.title('Gradient 64')
plt.imshow(img_3, cmap='gray')
plt.axis(False)
plt.subplot(1, 3, 2)
plt.title('Gradient uint8')
plt.imshow(img_4, cmap='gray')
plt.axis(False)
plt.subplot(1, 3, 3)
img4=cv2.absdiff(img_3,img_4) #negative slope
plt.title('Gradient sub')
plt.imshow(img4,cmap='gray')
plt.axis(False)
plt.show()
plt.subplot(1, 2, 1)
plt.title('gradient64')
plt.imshow(img_3 ,cmap='gray')
plt.axis(False)
plt.subplot(1, 2, 2)
plt.imshow(img_4 ,cmap='gray')
plt.title('gradient uint8')
plt.axis(False)
plt.show()



####################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lung.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
row, col = img.shape[:2]
median = cv2.medianBlur(img, 3)
kernel1_size = 3
img3 = np.zeros((row + (kernel1_size - 1), col + (kernel1_size - 1)), dtype=np.uint8)
img3[int(kernel1_size / 2) + 1:row + int(kernel1_size / 2) + 1,
int(kernel1_size / 2) + 1:col + int(kernel1_size / 2) + 1] = median[:, :]
#sobel_kernel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_kernel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y2=np.array([[-1, 2, -1], [-1,2, -1], [-1,2, -1]])
img_grad = np.zeros((row, col), dtype=np.int16)
img_grad2 = np.zeros((row, col), dtype=np.int16)
for i in range(0, row):
    for j in range(0, col):
        #cx = np.sum(np.multiply(img3[i:i + kernel1_size, j:j + kernel1_size], sobel_kernel_x))
        cy = np.sum(np.multiply(img3[i:i + kernel1_size, j:j + kernel1_size], sobel_kernel_y))
        cy2 = np.sum(np.multiply(img3[i:i + kernel1_size, j:j + kernel1_size], sobel_kernel_y2))
        img_grad[i][j] = np.abs(cy)
        img_grad2[i][j]=np.abs(cy2) #negative gradient
img_grad11=np.uint8((img_grad))
img_grad22=np.uint8((img_grad2))
plt.subplot(131)
plt.imshow(img_grad11, cmap='gray')
plt.title('Gradiented image 1')
plt.xticks([])
plt.yticks([])
plt.subplot(132)
plt.imshow(img_grad22, cmap='gray')
plt.title('Gradiented image 2')
plt.xticks([])
plt.yticks([])
plt.subplot(133)
plt.imshow(img, cmap='gray')
plt.title('original')
plt.xticks([])
plt.yticks([])
plt.show()




