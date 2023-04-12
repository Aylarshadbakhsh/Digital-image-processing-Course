import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('chest.tif',cv2.IMREAD_GRAYSCALE)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

phase_spectrum = cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1])
magnitude_spectrum = 20*np.log(1+np.abs(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
magnitude_spectrum_without_ffts = 20*np.log(1+np.abs(cv2.magnitude(dft[:,:,0],dft[:,:,1])))

plt.figure()
plt.subplot(221)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
plt.subplot(222)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')
plt.xticks([])
plt.yticks([])
plt.subplot(223)
plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('Phase Spectrum')
plt.xticks([])
plt.yticks([])
plt.subplot(224)
plt.imshow(magnitude_spectrum_without_ffts, cmap = 'gray')
plt.title('Magnitude Spectrum without fftshift')
plt.xticks([])
plt.yticks([])
plt.show()

#Without Using fftshift and by multiplying the image by (-1)^(x+y)
img2 = img.copy()
img2 = img2.astype(np.float32)

rows, cols = img.shape
for i in range(0,rows):
    for j in range(0,cols):
        img2[i][j]=np.power(-1,(i+j))*img[i][j]

dft2 = cv2.dft(np.float32(img2),flags = cv2.DFT_COMPLEX_OUTPUT)
phase_spectrum2 = cv2.phase(dft2[:,:,0],dft2[:,:,1])
magnitude_spectrum2 = 20*np.log(1+np.abs(cv2.magnitude(dft2[:,:,0],dft2[:,:,1])))
#Part2
img_back = cv2.idft(dft)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
plt.figure()
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(img_back, cmap = 'gray')
plt.title('Image with idft')
plt.xticks([]), plt.yticks([])
plt.show()

#Part3
dft2[:, :, 1] *=-1
img_combined = cv2.idft(dft2)
magnitude2,_= cv2.cartToPolar(img_combined[:, :, 0],img_combined[:, :, 1])
plt.figure()
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude2, cmap = 'gray')
plt.title('Mirrored Image')
plt.xticks([])
plt.yticks([])
plt.show()
