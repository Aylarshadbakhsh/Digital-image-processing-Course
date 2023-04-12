import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy  import array

img = cv2.imread('retina.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('retina_sub.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
print( 'image size',img.shape) #size of image(width and hieght)
print('type',img.dtype)#type of pixel
print('memory space',img.shape[0]*img.shape[1]*3*8)
plt.hist(img.ravel(),bins=256)
plt.title('histogram of  retina image ')
plt.show()
histg2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
resizeimg = cv2.resize(img, dsize=(565, 565), interpolation=cv2.INTER_LINEAR) #as same as retina_sub size
mask = np.zeros(img.shape[:2], np.uint8)
for i in range(0, img.shape[0],100):
   for j in range(0,img.shape[1],100):
      mask[i:i + 565,j:j + 565] = 255
      masked_img = cv2.bitwise_and(img, img, mask=mask)
      masked_img = masked_img[i:i + 565, j:j + 565]
      hist_1, _ = np.histogram(masked_img, bins=100, range=[0, 256])
      hist_2, _ = np.histogram(img2, bins=100, range=[0, 256])
      min = np.minimum(hist_1, hist_2)
      intersection = np.true_divide(np.sum(min), np.sum(hist_2))
      if(intersection>0.98):
       plt.imshow(masked_img,cmap='gray')
       plt.title('masked with intersection')
       plt.axis(False)
       plt.show()
for i in range(0, img.shape[0],100):
   for j in range(0,img.shape[1],100):
      mask[i:i + 565, j:j + 565] = 255
      masked_img = cv2.bitwise_and(img, img, mask=mask)
      masked_img = masked_img[i:i + 565, j:j + 565]
      histg1 = cv2.calcHist([masked_img], [0], None, [256], [0, 256])
      histg2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
      if (cv2.compareHist(histg1, histg2, cv2.HISTCMP_CORREL) >0.99):
          hist_full1 = cv2.calcHist([resizeimg], [0], None, [256], [0, 256])
          hist_full = cv2.calcHist([img2], [0], None, [256], [0, 256])
          hist_mask = cv2.calcHist([masked_img], [0], None, [256], [0, 256])
          plt.bar(range(0, 256), hist_full[:, 0])
          plt.title('histogram of retina-sub')
          plt.show()
          plt.bar(range(0, 256), hist_mask[:, 0])
          plt.title('histogram of MASK')
          plt.show()
          plt.bar(range(0, 256), hist_full1[:, 0])
          plt.title('histogram of retina')
          plt.show()
          plt.subplot(221), plt.imshow(resizeimg, 'gray'),plt.title('resized image')
          plt.axis(False)
          plt.subplot(222), plt.imshow(masked_img, 'gray'), plt.title('masked')
          plt.axis(False)
          plt.subplot(223), plt.imshow(img2, 'gray'), plt.title('retina-sub')
          plt.axis(False)
          plt.show()
          plt.imshow(masked_img, 'gray')
          plt.title('masked with cv2.comparehist')
          plt.axis(False)
          plt.show()


def Binary(img1=[]):
    img1 = np.array(img)
    row, col = img1.shape[:2]
    binary = np.array(np.zeros((row, col, 8), dtype=np.uint8))
    for i in range(0, row):
        for j in range(0, col):
            x = (format(img1[i][j], '08b'))
            for k in range(0, 8):
                binary[i][j][k] = x[7 - k]
    return binary


binary = Binary(img)
plt.figure()
plt.subplot(241)
plt.imshow(binary[:,:,0],cmap ='gray')
plt.title('2^0')
plt.xticks([])
plt.yticks([])
plt.subplot(242)
plt.imshow(binary[:,:,1],cmap ='gray')
plt.title('2^1')
plt.xticks([])
plt.yticks([])
plt.subplot(243)
plt.imshow(binary[:,:,2] ,cmap='gray')
plt.title('2^2')
plt.xticks([])
plt.yticks([])
plt.subplot(244)
plt.imshow(binary[:,:,3] ,cmap='gray')
plt.title('2^3')
plt.xticks([])
plt.yticks([])
plt.subplot(245)
plt.imshow(binary[:,:,4],cmap ='gray')
plt.title('2^4')
plt.xticks([])
plt.yticks([])
plt.subplot(246)
plt.imshow(binary[:,:,5],cmap ='gray')
plt.title('2^5')
plt.xticks([])
plt.yticks([])
plt.subplot(247)
plt.imshow(binary[:,:,6] ,cmap='gray')
plt.title('2^6')
plt.xticks([])
plt.yticks([])
plt.subplot(248)
plt.imshow(binary[:,:,7] ,cmap='gray')
plt.title('2^7')
plt.xticks([])
plt.yticks([])
plt.show()



