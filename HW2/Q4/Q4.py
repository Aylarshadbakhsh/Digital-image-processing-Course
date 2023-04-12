import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
img=cv2.imread('bone-scan.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(img, 7)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
plt.yticks([])
plt.xticks([])
plt.title('median')
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.yticks([])
plt.xticks([])
plt.show()
figure_size =7
mean = cv2.blur(img,(figure_size, figure_size))
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
plt.yticks([])
plt.xticks([])
plt.title('median')
plt.yticks([])
plt.xticks([])
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(mean, cv2.COLOR_BGR2RGB))
plt.title('mean')
plt.yticks([])
plt.xticks([])
plt.show()
row , col = img.shape[:2]
la_kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel1_size = 3
img3 = np.zeros((row + (kernel1_size - 1), col + (kernel1_size - 1)), dtype=np.uint8)
img3[int(kernel1_size/2)+1:row+int(kernel1_size/2)+1,int(kernel1_size/2)+1:col+int(kernel1_size/2)+1]=median[:,:]


img_la1 = np.zeros((row, col), dtype=np.int8)

for i in range(0, row):
    for j in range(0, col):
        img_la1[i][j] = np.sum(np.multiply(img3[i:i + kernel1_size, j:j + kernel1_size], la_kernel1))


cv2.imshow('laplacian',img_la1)
cv2.waitKey()
cv2.imwrite('laplacian.jpg',img_la1)
c0 = 1
delta_c= 0.5
x11=(c0*img_la1+median)
#x11=x11/np.max(x11)*255 #contrast stretching 0-255
b1 = np.array(np.zeros((row, col), dtype=np.uint8))

for i in range(0, row):#mapping <0 to 0 and 255> to 255
    for j in range(0, col):

        if (x11[i][j] >= 255):
            b1[i][j] = 255
            if (x11[i][j] < 0):
                b1[i][j] = 0
            else:
                b1[i][j] = x11[i][j]

b1 = b1.astype(np.uint8)
l=plt.imshow(x11,cmap='gray')
plt.xticks([])
plt.yticks([])
axcla = plt.axes([0.25, 0.1, 0.65, 0.03])
scla = Slider(axcla,'c', -20, 20, valinit=c0, valstep=delta_c)

def update(val):
    c=scla.val
    l.set_data(c*img_la1+median)
    fig.canvas.draw_idle()
scla.on_changed(update)
plt.show()
