import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

img = cv2.imread('brains.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
row, col = img.shape[:2]
a2 = np.array(np.zeros((row, col), dtype=np.uint8))
a3 = np.array(np.zeros((row, col), dtype=np.uint8))

#HISTOGAM fuction
'''def Histogram(img=[]):
    img = np.array(img)
    row, col = img.shape[:2]
    count = []
    for i in range(0, 256):
        count.append(np.sum(img == i))
    return count
counta2=Histogram(a2)
npa1 = np.asarray(counta2, dtype=np.float32)
np1= np.reshape(npa1, (256, 1))
plt.figure(4)
plt.bar(range(0, 256), np1[:, 0])
plt.show()'''


def tran1(img):
 #c=5
 log_transformed = 5*np.log(1+img)
 a2 = np.array(log_transformed, dtype=np.uint8)
 a2 = np.round(a2 / np.max(a2) * 255)#contrast stretching
 a2 = a2.astype(np.uint8)
 return  a2
gamma=0.25
def tran2(img):
 #c=3.5
 a3= np.array( 3.5*255 * (img / 255) ** gamma, dtype='uint8')  #powertranformed with contrast stretching
 return a3
a3=tran2(img)
a2=tran1(img)
plt.figure()
plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(a2, cv2.COLOR_BGR2RGB))
plt.title('logtransformed')
plt.xticks([])
plt.yticks([])
plt.subplot(2,3,3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('original')
plt.xticks([])
plt.yticks([])
plt.subplot(2,3,2)
plt.imshow(cv2.cvtColor(a3, cv2.COLOR_BGR2RGB))
plt.title('powertransformed')
plt.xticks([])
plt.yticks([])
histg2 = cv2.calcHist([a2], [0], None, [256], [0, 256])
plt.subplot(2,3,4)
plt.bar(range(0, 256), histg2[:, 0])
plt.yticks([])
histgimg1 = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.subplot(2,3,6)
plt.bar(range(0,256),histgimg1[:, 0])
plt.yticks([])
plt.subplot(2,3,5)
histg3 = cv2.calcHist([a3], [0], None, [256], [0, 256])
plt.bar(range(0,256),histg3[:, 0])
plt.yticks([])
plt.show()


