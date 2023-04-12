
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('xray_checkered.png',cv2.IMREAD_GRAYSCALE)
img = np.array(img)
rows, cols = img.shape
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_spectrum1 = 20*np.log(1+np.abs(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])))
phase_spectrum = cv2.phase(dft_shift[:, :, 1], dft_shift[:, :, 0])
for i in range(120, 129):#left
    for j in range(63, 70):
        magnitude_spectrum[i][j]=(magnitude_spectrum[i-1][j]+magnitude_spectrum[i+1][j]+magnitude_spectrum[i][j+1]+magnitude_spectrum[i][j-1])/8 #/4
    for j in range(184, 190):#right
        magnitude_spectrum[i][j] = (magnitude_spectrum[i - 1][j] + magnitude_spectrum[i + 1][j] +
                                            magnitude_spectrum[i][j + 1] + magnitude_spectrum[i][j - 1])/8


for i in range(182, 189):#down
    for j in range(124, 131):
              magnitude_spectrum[i][j]=(magnitude_spectrum[i-1][j]+magnitude_spectrum[i+1][j]+magnitude_spectrum[i][j+1]+magnitude_spectrum[i][j-1])/8


for i in range(63,70):#up
    for j in range(124, 131):
        magnitude_spectrum[i][j] = (magnitude_spectrum[i - 1][j] + magnitude_spectrum[i + 1][j] + magnitude_spectrum[i][
            j + 1] + magnitude_spectrum[i][j - 1]) /8
'''plt.imshow(magnitude_spectrum1,cmap='gray')
plt.show()'''
magnitude_spectrum2=np.log(magnitude_spectrum) / 20
'''plt.imshow(magnitude_spectrum2,cmap='gray')
plt.show()'''
img_dft2 = np.zeros((rows, cols,2), dtype=np.float32)
img_dft2[:, :, 0], img_dft2[:, :, 1] = cv2.polarToCart(magnitude_spectrum, phase_spectrum)
img_combined2 = cv2.idft(img_dft2)
magnitude, _ = cv2.cartToPolar(img_combined2[:, :, 0], img_combined2[:, :, 1])
result = np.zeros((rows, cols), dtype=np.float32)
result[:, :] = magnitude[0:rows, 0:cols]
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(result, cmap='gray')
plt.title('denoised image')
plt.xticks([]), plt.yticks([])
plt.show()
