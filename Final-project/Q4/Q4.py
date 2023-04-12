import cv2
import numpy as np
import matplotlib.pyplot as plt

m = int(input("enter n(if u choose 1 it means with 8 neighborhood and if u choose 0 it means 4 neighborhood "))
if m == 0 or m == 1:
    if m == 0:
        def getn(x, y, shape):  # 4neighborhood (0.1),(0.-1),(-1,0),(1,0)
            out = []
            maxx = shape[1] - 1
            maxy = shape[0] - 1
            outx = x
            outy = min(max(y - 1, 0), maxy)
            out.append((outx, outy))
            outx = min(max(x - 1, 0), maxx)
            outy = y
            out.append((outx, outy))
            outx = min(max(x + 1, 0), maxx)
            outy = y
            out.append((outx, outy))
            outx = x
            outy = min(max(y + 1, 0), maxy)
            out.append((outx, outy))
            return out


        def regiongrowing(img, seed):
            outimg = np.zeros_like(img)
            list = []
            list2 = []
            list.append((seed[0], seed[1]))
            while (len(list) > 0):
                pix = list[0]
                outimg[pix[0], pix[1]] = 255
                for n in getn(pix[0], pix[1], img.shape):
                    if np.abs(img[n[0], n[1]] - img[seed[0], seed[1]]) <= 11:
                        outimg[n[0], n[1]] = 255
                        if not n in list2:
                            list.append(n)
                        list2.append(n)
                list.pop(0)
            return outimg


        def on_mouse(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                print('Seed: ' + str(x) + ', ' + str(y), img[y, x], clicks.append((y, x)))


        clicks = []
        image = cv2.imread('FMRI.jpg', 0)
        img = image
        cv2.namedWindow('Input')
        cv2.setMouseCallback('Input', on_mouse, 0, )
        cv2.imshow('Input', img)
        cv2.waitKey()
        seed = clicks[-1]
        out = regiongrowing(img, seed)
        cv2.imshow('4 neighbourhood ', out)
        cv2.waitKey()
        cv2.destroyAllWindows()
        for i in range(0, out.shape[0]):
            for j in range(0, out.shape[1]):
                if out[i][j] == 255:
                    image[i][j] = 255
        plt.imshow(image, cmap='gray')
        plt.title('4 neighborhood')
        plt.axis(False)
        plt.show()
        plt.imsave('out4.png', image, cmap='gray')
    else:
        def getn(x, y, shape):  # 8neighborhood (0.1),(0.-1),(-1,0),(1,0),(1,1),(-1,1),(1,-1),(-1,-1)
            out = []
            maxx = shape[1] - 1
            maxy = shape[0] - 1
            outx = min(max(x - 1, 0), maxx)
            outy = min(max(y - 1, 0), maxy)
            out.append((outx, outy))
            outx = x
            outy = min(max(y - 1, 0), maxy)
            out.append((outx, outy))
            outx = min(max(x + 1, 0), maxx)
            outy = min(max(y - 1, 0), maxy)
            out.append((outx, outy))
            outx = min(max(x - 1, 0), maxx)
            outy = y
            out.append((outx, outy))
            outx = min(max(x + 1, 0), maxx)
            outy = y
            out.append((outx, outy))
            outx = min(max(x - 1, 0), maxx)
            outy = min(max(y + 1, 0), maxy)
            out.append((outx, outy))
            outx = x
            outy = min(max(y + 1, 0), maxy)
            out.append((outx, outy))
            outx = min(max(x + 1, 0), maxx)
            outy = min(max(y + 1, 0), maxy)
            out.append((outx, outy))
            return out


        def regiongrowing(img, seed):
            outimg = np.zeros_like(img)
            list = []
            list2 = []
            list.append((seed[0], seed[1]))
            while (len(list) > 0):
                pix = list[0]
                outimg[pix[0], pix[1]] = 255
                for n in getn(pix[0], pix[1], img.shape):
                    if np.abs(img[n[0], n[1]] - img[seed[0], seed[1]]) <= 11:
                        outimg[n[0], n[1]] = 255
                        if not n in list2:
                            list.append(n)
                        list2.append(n)
                list.pop(0)
            return outimg


        def on_mouse(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                print('Seed: ' + str(x) + ', ' + str(y), img[y, x], clicks.append((y, x)))


        clicks = []
        image = cv2.imread('FMRI.jpg', 0)
        img = image
        cv2.namedWindow('Input')
        cv2.setMouseCallback('Input', on_mouse, 0, )
        cv2.imshow('Input', img)
        cv2.waitKey()
        seed = clicks[-1]
        out = regiongrowing(img, seed)
        cv2.imshow('8 neighbourhood  ', out)
        cv2.waitKey()
        for i in range(0, out.shape[0]):
            for j in range(0, out.shape[1]):
                if out[i][j] == 255:
                    image[i][j] = 255
        plt.imshow(image, cmap='gray')
        plt.title('8neighborhood')
        plt.axis(False)
        plt.show()
        plt.imsave('out8.png',image,cmap='gray')
else:
      print('invalid ,please enter 1 or 0')