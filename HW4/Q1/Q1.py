import numpy as np
import matplotlib.pyplot as plt
import cv2
from tools import *
from skimage import restoration
img=cv2.imread('Retina.jpg',cv2.IMREAD_GRAYSCALE)
#part 1
kernel=1/13*np.eye(13,13)
motion_blurred= cv2.filter2D(img, -1, kernel)
plt.imshow(motion_blurred,cmap='gray')
plt.axis(False)
plt.title('motion blurred')
plt.show()
#PART 2
batch_image=normal(motion_blurred) #converting to [-1,1] for wiener filter's input
output1 = restoration.wiener(batch_image,kernel,0.0001)
output2 = restoration.wiener(batch_image,kernel,0.00009)
restored1= n_range(output1)#converting to [0-255] to display image
restored2= n_range(output2)
plt.imshow(restored1,cmap='gray')
plt.axis(False)
plt.show()
#part 3
fft_restored1=logmagnitude(restored1)
fft_restored2=logmagnitude(restored2)
fftimg=logmagnitude(img)
plt.figure()
plt.subplot(2,2,1)
plt.imshow(restored1,cmap='gray',vmin=0,vmax=255)
plt.title('restored')
plt.axis(False)
plt.subplot(2,2,3)
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.axis(False)
plt.title('original')
plt.subplot(2,2,2)
plt.imshow(fft_restored1,cmap='gray')
plt.title('magnitude spectrum of restored')
plt.axis(False)
plt.subplot(2,2,4)
plt.title('magnitude spectrum of original')
plt.imshow(fftimg,cmap='gray')
plt.axis(False)
plt.show()
plt.figure()
plt.subplot(1,2,1)
plt.imshow(restored1,cmap='gray')
plt.title('restored')
plt.axis(False)
plt.subplot(1,2,2)
plt.imshow(img,cmap='gray')
plt.axis(False)
plt.title('original')
plt.show()
import numpy as np
from numpy.fft import fft2, ifft2
#manual code f0r wiener filter
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    a = np.copy(img)
    a= fft2(a)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    a = a* kernel
    a = np.abs(ifft2(a))
    return a
filtered_img1 = wiener_filter(motion_blurred, kernel, K =0.0001)
plt.subplot(2,2,1)
plt.imshow(restored1,cmap='gray')
plt.axis(False)
plt.title('RESTORED1')
plt.subplot(2,2,2)
plt.imshow(restored2,cmap='gray')
plt.axis(False)
plt.title('RESTORED2')
plt.subplot(2,2,3)
plt.imshow(fft_restored1,cmap='gray')
plt.axis(False)
plt.subplot(2,2,4)
plt.imshow(fft_restored2,cmap='gray')
plt.axis(False)
plt.show()
plt.subplot(1,2,1)
plt.imshow(restored1,cmap='gray')
plt.axis(False)
plt.title('RESTORED1')
plt.subplot(1,2,2)
plt.imshow(restored2,cmap='gray')
plt.title('RESTORED2')
plt.axis(False)
plt.show()