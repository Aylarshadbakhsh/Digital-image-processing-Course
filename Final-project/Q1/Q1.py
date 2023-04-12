import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('noisy_skull.png',0)
row,col=img.shape[:2]
print(row,col)
pad=cv2.copyMakeBorder(img, 0, 1,4, 4, cv2.BORDER_REPLICATE) #padding
noisy=img[:81,:81]
var2 = np.var(noisy)
mean2=np.mean(noisy)
print('mean of noise',mean2)
print('variance of noise',var2)
def Histogram(img=[]):
    img = np.array(img)
    row, col = img.shape[:2]
    count = []
    for i in range(0, 256):
        count.append\
            (np.sum(img == i))
    return count
count1=Histogram(img)
count2=Histogram(noisy)
def normalize(count_d,row_d,col_d):
 npa1 = np.asarray(count_d, dtype=np.float32)
 np1= np.reshape(npa1, (256, 1))
 np1=np1/(row_d*col_d)
 return np1
np1=normalize(count1,row,col)
np2=normalize(count2,81,81)
plt.subplot(1,2,1)
plt.bar(range(0, 256),np1[:,0])
plt.title('Normalized Histogram of image ')
plt.subplot(1,2,2)
plt.bar(range(0, 256), np2[:,0])
plt.title('Normalized Histogram of noise')
plt.show()
mask = np.zeros(pad.shape[:2], np.uint8)
mask4 = np.zeros(pad.shape[:2], np.uint8)
for i in range(0, pad.shape[0],9):#adaptive noise filter
   for j in range(0,pad.shape[1],9):
       mask[i:i + 9,  j:j + 9] = 255#window to slide over image and finding mean and variance of window
       masked_img = cv2.bitwise_and(pad, pad, mask=mask)
       masked_img=masked_img[i:i + 9,  j:j + 9] #evey iteration,new window"s size  must be 9*9
       var1 = np.var(masked_img)
       mean = np.mean(masked_img)
       mask2 = masked_img - ((var2 / var1 )* (masked_img - mean))
       mask4[i:i+9,j:j+9]=mask2 
#print(mask4.dtype)
plt.subplot(1,2,1)
plt.imshow(mask4, cmap='gray',vmin=0,vmax=255)
plt.title('adaptive4')
plt.axis(False)
plt.subplot(1,2,2)
median=cv2.medianBlur(img,7)
plt.imshow(median ,cmap='gray',vmin=0,vmax=255)
plt.title('median')
plt.axis(False)
plt.show()

plt.subplot(121)
plt.imshow(mask4, cmap='gray',vmin=0,vmax=255)
plt.title('adaptive')
plt.axis(False)
plt.subplot(122)
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.title('original')
plt.axis(False)
plt.show()
median=cv2.medianBlur(img,7)
mediannoise=median[0:81,0:81]
countmediannoise=Histogram(mediannoise)
mask4noise=mask4[0:81,0:81]
countmask=Histogram(mask4noise)
npmed=normalize(countmediannoise,81,81)
npmask=normalize(countmask,81,81)
plt.subplot(231),plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.title('original image'),plt.axis(False)
plt.subplot(234),plt.bar(range(0, 256),np2[:,0])
plt.subplot(232),plt.imshow(mask4,cmap='gray')
plt.title('adaptive'),plt.axis(False)
plt.subplot(235),plt.bar(range(0, 256),npmask[:,0])
plt.subplot(233),plt.imshow(median,cmap='gray',vmin=0,vmax=255)
plt.title('median'),plt.axis(False)
plt.subplot(236),plt.bar(range(0, 256),npmed[:,0])
plt.show()