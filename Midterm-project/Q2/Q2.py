import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
img1=cv2.imread('CT_1.tif',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('CT_2.tif',cv2.IMREAD_GRAYSCALE)
row, col = img1.shape[:2]
a3 = np.array(np.zeros((row, col), dtype=np.uint8))
def tran1(img):
 a=np.pi/(255*2) #a<pi /r*2 rmax=255  a<pi/255*2
 sin1 = np.sin(a*img)*(256 - 1)
 a3= np.array(sin1, dtype=np.uint8)
 a3 = np.round(a3 / np.max(a3) * 255)#contrast stretching
 a3= a3.astype(np.uint8)
 return a3
a1=tran1(img1)
a2=tran1(img2)
plt.figure()
plt.subplot(2,2,1)
plt.imshow(a1,cmap='gray',vmin=0,vmax=255)
plt.title('transformed image CT1')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.imshow(img1,cmap='gray',vmin=0,vmax=255)
plt.title('original image CT1')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
plt.imshow(a2,cmap='gray',vmin=0,vmax=255)
plt.title('tarnsformed CT2')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,4)
plt.imshow(img2,cmap='gray',vmin=0,vmax=255)
plt.title('original CT2')
plt.xticks([])
plt.yticks([])
plt.show()
x=np.arange(0,255,1)
y=x
r=np.arange(0,np.max(img1),1)
s=255*np.sin(r*np.pi/510)
plt.show()
plt.figure()
w=np.arange(0,np.max(img2),1)
q=255*np.sin(w*np.pi/510)
t=np.arange(0,255,1)
u=255-t
plt.plot(w,q,c="r",label="transformed function  CT2")
plt.plot(t,u,c="b",label=" negative")
plt.plot(r,s,c="g",label="transformed function CT1")
plt.plot(x,y,c='orange',label='identity function')
plt.legend()
plt.show()