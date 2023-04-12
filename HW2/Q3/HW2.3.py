
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def Histogram(img=[]):
    img = np.array(img)
    row, col = img.shape[:2]
    count = []
    for i in range(0, 256):
        count.append(np.sum(img == i))
    return count

img_d= cv2.imread('Dark.tif')
img_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2GRAY)
row_d, col_d = img_d.shape[:2]


img_l = cv2.imread('Lowcontrast.tif')
img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
row_l, col_l = img_l.shape[:2]

img_b = cv2.imread('Bright.tif')
img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
row_b, col_b = img_b.shape[:2]


count_d = Histogram(img_d)
count_b = Histogram(img_b)
count_l = Histogram(img_l)
def normalize(count_d,row_d,col_d):
 npa1 = np.asarray(count_d, dtype=np.float32)
 np1= np.reshape(npa1, (256, 1))
 np1=np1/(row_d*col_d)
 return np1
np1=normalize(count_d,row_d,col_d)
np2=normalize(count_l,row_l,col_l)
np3=normalize(count_b,row_b,col_b)
plt.figure()
plt.bar(range(0, 256),np1[:,0])
plt.title('Normalized Histogram dark')
plt.figure()
plt.bar(range(0, 256),np2[:,0])
plt.title('Normalized Histogram lowcontrast')
plt.figure()
plt.bar(range(0, 256),np3[:,0])
plt.title('Normalized Histogram bright')
plt.show()
def equalize(histg,img_b,row_b,col_b):
    H = np.round(np.cumsum(histg) * (np.size(histg) - 1))
    new_hist = []
    for i in range(0, np.size(histg)):
        new_hist.append(sum(histg[np.where(H == i)]))
    img_r_b = np.zeros((row_b, col_b), dtype=np.uint8)
    for i in range(0, row_b):
        for j in range(0, col_b):
            img_r_b[i][j] = H[img_b[i][j]]

    return new_hist, img_r_b

img_r_b = np.zeros((row_b, col_b), dtype=np.uint8)
img_r_d = np.zeros((row_d, col_d), dtype=np.uint8)
img_r_l = np.zeros((row_l, col_l), dtype=np.uint8)
he_b= []
he_d = []
he_l = []
he_b, img_r_b = equalize(np3, img_b,row_b,col_b)
he_d,img_r_d =equalize(np1,img_d,row_d,col_d)
he_l,img_r_l=equalize(np2,img_l,row_l,col_l)

f = plt.figure(1)
f.add_subplot(2,2,1)
plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
f.add_subplot(2,2,3)
plt.bar(range(0, 256), count_b)
plt.title('bright')
f.add_subplot(2,2,2)
plt.imshow(img_r_b, cmap='gray')
plt.xticks([])
plt.yticks([])
f.add_subplot(2,2,4)
plt.bar(range(0, 256), he_b)
plt.title('Equalized Histogram bright')
plt.show()


f1=plt.figure(2)
f1.add_subplot(2,2,1)
plt.imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
f1.add_subplot(2,2,3)
plt.bar(range(0, 256), count_d)
plt.title('dark')
f1.add_subplot(2,2,2)
plt.imshow(img_r_d, cmap='gray')
plt.xticks([])
plt.yticks([])
f1.add_subplot(2,2,4)
plt.bar(range(0, 256), he_d)
plt.title('Equalized Histogram  dark')
plt.show()
f1=plt.figure(3)
f1.add_subplot(2,2,1)
plt.imshow(cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
f1.add_subplot(2,2,3)
plt.bar(range(0, 256), count_l)
plt.title('lowcontrast')
f1.add_subplot(2,2,2)
plt.imshow(img_r_l, cmap='gray')
plt.xticks([])
plt.yticks([])
f1.add_subplot(2,2,4)
plt.bar(range(0, 256), he_l)
plt.title('Equalized Histogram  lowcontarst')
plt.show()