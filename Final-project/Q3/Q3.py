import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture('Q_three.avi')
frames = []

while (video.isOpened()):

    success, image = video.read()
    if success == False:
        break
    frames.append(image)

x = frames[0].shape
color = []
gray = []
for i in range(0, len(frames)):
    color.append(cv2.cvtColor(np.array(frames[i]), cv2.COLOR_BGR2RGB))
    gray.append(cv2.cvtColor(np.array(frames[i]), cv2.COLOR_BGR2GRAY))

AVG = np.array(np.zeros((x[0], x[1]), dtype=np.float64))

for i in range(0, len(frames)):
    AVG += gray[i]
    #print(len(frames))

AVG /= len(frames)

plt.figure()
plt.imshow(AVG, cmap='gray')
plt.title('Mean')
plt.xticks([])
plt.yticks([])
plt.show()

cars = []
for i in range(0, len(frames)):
    cars.append(abs(gray[i] - AVG))

plt.figure()
plt.imshow(cars[92], cmap='gray')
plt.title('Splited Cars')
plt.xticks([])
plt.yticks([])
plt.show()

ther = []
for i in range(0, len(frames)):
    ret, thresh = cv2.threshold(cars[i], 45, 1, cv2.THRESH_BINARY)
    ther.append(thresh)

plt.figure()
plt.imshow(ther[92], cmap='gray')
plt.title('Thereshold Splited Cars')
plt.xticks([])
plt.yticks([])
plt.show()

output = []
for i in range(0, len(frames)):
    CF = color[i]
    CR = CF[:, :, 0]
    CF[:, :, 0] = CR + (255 - CR) * ther[i]
    CG = CF[:, :, 1]
    CF[:, :, 1] = CG - CG * ther[i]
    CB = CF[:, :, 2]
    CF[:, :, 2] = CB - CB * ther[i]
    output.append(CF)

plt.figure()
plt.imshow(output[92], cmap='gray')
plt.title('Final Output')
plt.xticks([])
plt.yticks([])
plt.show()

for i in range(0, len(frames)):
    output[i] = cv2.cvtColor(np.array(output[i]), cv2.COLOR_RGB2BGR)

height, width, _ = output[0].shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("Output.avi", fourcc, 24.0, (width, height))

for i in range(0, len(frames)):
    video.write(output[i])

cv2.destroyAllWindows()
video.release()