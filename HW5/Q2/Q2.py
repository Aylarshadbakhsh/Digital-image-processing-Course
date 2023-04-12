import cv2
import numpy as np
import matplotlib.pyplot as plt


def split(img, n, x, y, r, roi):
    if (r == n):
        roi.append([x, y, r])
        return roi
    else:
        if (np.mean(img[x:x + r, y:y + r]) >= 70 and np.mean(img[x:x + r, y:y + r]) <= 150):
            roi.append([x, y, r])
            return roi
        else:
            split(img, n, x, y, int(r / 2), roi)
            split(img, n, x + int(r / 2), y, int(r / 2), roi)
            split(img, n, x, y + int(r / 2), int(r / 2), roi)
            split(img, n, x + int(r / 2), y + int(r / 2), int(r / 2), roi)
            return roi


img = cv2.imread('fMRI.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Gray = x = gray[2:514, 2:514]


quadReg = [16, 8, 4, 2]
for j in range(0, 4):

    roi = split(Gray, quadReg[j], 0, 0, 512, [])
    result = np.zeros((512, 512), dtype=np.uint8)

    # Merge
    for i in roi:
        if (np.mean(Gray[i[0]:i[0] + i[2], i[1]:i[1] + i[2]]) >= 70 and np.mean(
                Gray[i[0]:i[0] + i[2], i[1]:i[1] + i[2]]) <= 150):
            result[i[0]:i[0] + i[2], i[1]:i[1] + i[2]] = 255

    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.title('Result (minimum quadregion = {} * {})'.format(quadReg[j], quadReg[j]))
    plt.axis(False)
    plt.show()