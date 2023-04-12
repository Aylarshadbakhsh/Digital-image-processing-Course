#type: #1:ideal 2:gaussian 3:Butterworth
# kind: 1=lpf 2=HPF
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('a.tif',cv2.IMREAD_GRAYSCALE)
def function1(Type, kind, D0, n, img=[]):
    img = np.array(img)
    rows, cols = img.shape
    padded_img = np.zeros((rows * 2, cols * 2), dtype=np.float32)
    padded_img[0:rows, 0:cols] = img[:, :]
    dft = cv2.dft(np.float32(padded_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    phase_spectrum = cv2.phase(dft_shift[:, :, 1], dft_shift[:, :, 0])
    result_fft = np.zeros((rows * 2, cols * 2), dtype=np.float32)

    for i in range(0, rows * 2):
        for j in range(0, cols * 2):
            D = np.sqrt(np.power((i - rows), 2) + np.power((j - cols), 2))
            if (kind == 1): #lpf
                if (Type == 1):
                    if (D <= D0):
                        result_fft[i][j] = magnitude_spectrum[i][j]
                if (Type == 2):
                    result_fft[i][j] = np.exp(-(D * D) / (2 * D0 * D0)) * magnitude_spectrum[i][j]
                if (Type == 3):
                    result_fft[i][j] = (1 / (1 + np.power((D / D0), 2 * n))) * magnitude_spectrum[i][j]
            else: #HPF
                if (Type == 1):
                    if (D >= D0):
                        result_fft[i][j] = magnitude_spectrum[i][j]
                if (Type == 2):
                    result_fft[i][j] = (1 - np.exp(-(D * D) / (2 * D0 * D0))) * magnitude_spectrum[i][j]
                if (Type == 3):
                    result_fft[i][j] = (1 / (1 + np.power((D0 / D), 2 * n))) * magnitude_spectrum[i][j]

    img_dft2 = np.zeros((rows * 2, cols * 2, 2), dtype=np.float32)
    img_dft2[:, :, 0], img_dft2[:, :, 1] = cv2.polarToCart(result_fft, phase_spectrum)
    img_combined2 = cv2.idft(img_dft2)
    magnitude, _ = cv2.cartToPolar(img_combined2[:, :, 0], img_combined2[:, :, 1])
    result = np.zeros((rows, cols), dtype=np.float32)
    result[:, :] = magnitude[0:rows, 0:cols]

    return result
#ideal type=1
magnitude111 = function1(1, 1, 50, 0, img)
magnitude112 = function1(1, 1, 100, 0, img)
magnitude113 = function1(1, 1, 200, 0, img)
magnitude121 = function1(1, 2, 50, 0, img)
magnitude122 = function1(1, 2, 100, 0, img)
magnitude123 = function1(1, 2, 200, 0, img)

plt.figure()
plt.subplot(231)
plt.imshow(magnitude111, cmap='gray')
plt.title('ILPF with D0 = 50')
plt.xticks([]), plt.yticks([])


plt.subplot(232)
plt.imshow(magnitude112, cmap='gray')
plt.title('ILPF with D0 =100')
plt.xticks([]), plt.yticks([])
plt.subplot(233)
plt.imshow(magnitude113, cmap='gray')
plt.title('ILPF with D0 = 200')
plt.xticks([]), plt.yticks([])
plt.subplot(234)
plt.imshow(magnitude121, cmap='gray')
plt.title('IHPF with D0 = 50')
plt.xticks([]), plt.yticks([])
plt.subplot(235)
plt.imshow(magnitude122, cmap='gray')
plt.title('IHPF with D0 = 100')
plt.xticks([]), plt.yticks([])
plt.subplot(236)
plt.imshow(magnitude123, cmap='gray')
plt.title('IHPF with D0 = 200')
plt.xticks([]), plt.yticks([])
plt.show()
# Gussian type=2
magnitude211 = function1(2, 1, 50, 0, img)
magnitude212 = function1(2, 1, 100, 0, img)
magnitude213 = function1(2, 1, 200, 0, img)
magnitude221 = function1(2, 2, 50, 0, img)
magnitude222 = function1(2, 2, 100, 0, img)
magnitude223 = function1(2, 2, 200, 0, img)

plt.figure()
plt.subplot(231)
plt.imshow(magnitude211, cmap='gray')
plt.title('GLPF with D0 = 50')
plt.xticks([])
plt.yticks([])
plt.subplot(232)
plt.imshow(magnitude212, cmap='gray')
plt.title('GLPF with D0 = 100')
plt.xticks([])
plt.yticks([])
plt.subplot(233)
plt.imshow(magnitude213, cmap='gray')
plt.title('GLPF with D0 = 200')
plt.xticks([]), plt.yticks([])
plt.subplot(234)
plt.imshow(magnitude221, cmap='gray')
plt.title('GHPF with D0 = 50')
plt.xticks([]), plt.yticks([])
plt.subplot(235)
plt.imshow(magnitude222, cmap='gray')
plt.title('GHPF with D0 = 100')
plt.xticks([])
plt.yticks([])
plt.subplot(236)
plt.imshow(magnitude223, cmap='gray')
plt.title('GHPF with D0 = 200')
plt.xticks([]), plt.yticks([])
plt.show()

# Butterworth type=3
magnitude311 = function1(3, 1, 50, 2.25, img)
magnitude312 = function1(3, 1, 100, 2.25, img)
magnitude313 = function1(3, 1, 200, 2.25, img)
magnitude321 = function1(3, 2, 50, 2.25, img)
magnitude322 = function1(3, 2, 100, 2.25, img)
magnitude323 = function1(3, 2, 200, 2.25, img)

plt.figure()
plt.subplot(231)
plt.imshow(magnitude311, cmap='gray')
plt.title('BLPF with D0 = 50')
plt.xticks([]), plt.yticks([])
plt.subplot(232)
plt.imshow(magnitude312, cmap='gray')
plt.title('BLPF with D0 = 100')
plt.xticks([]), plt.yticks([])
plt.subplot(233)
plt.imshow(magnitude313, cmap='gray')
plt.title('BLPF with D0 = 200')
plt.xticks([]), plt.yticks([])
plt.subplot(234)
plt.imshow(magnitude321, cmap='gray')
plt.title('BHPF with D0 =50')
plt.xticks([]), plt.yticks([])
plt.subplot(235)
plt.imshow(magnitude322, cmap='gray')
plt.title('BHPF with D0 = 100')
plt.xticks([]), plt.yticks([])
plt.subplot(236)
plt.imshow(magnitude323, cmap='gray')
plt.title('BHPF with D0 = 200')
plt.xticks([]), plt.yticks([])
plt.show()


