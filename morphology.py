import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import os

def loadGray2(img):
    image = cv2.imread(img, 0)
    return image


def displayImg(img, seconds = 5):
    cv2.imshow("Image", img)
    cv2.waitKey(1000 * seconds)
    cv2.destroyAllWindows()
    

image = cv2.imread("~HealthyImgs/1.tif")
displayImg(image, 50)

img = image

# Blurring 
gauss = cv2.GaussianBlur(img, (5, 5), 0)
gaussH = cv2.GaussianBlur(img, (17, 3), 0)
gaussB = cv2.GaussianBlur(img, (15, 15), 0) 
displayImg(img, 5)
displayImg(gauss, 5)
displayImg(gaussB, 5)
displayImg(gaussH, 5)


cv2.imwrite("~HealthyImgs/blur1.tif", gauss)
cv2.imwrite("~HealthyImgs/blur1H.tif", gaussH)

# horizintal rect kernel might be ideal
# gradients of image and blurs

# canny
one = 200
two = 2
gradient = cv2.Canny(img, one, two,L2gradient=True)

displayImg(gradient, 5)

# sobel
gradientSv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
gradientSvG = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=5)
gradientSvH = cv2.Sobel(gaussH, cv2.CV_64F, 0, 1, ksize=5)
gradientSvB = cv2.Sobel(gaussB, cv2.CV_64F, 0, 1, ksize=5)
displayImg(gradientSv, 5)
displayImg(gradientSvG, 5)
displayImg(gradientSvH, 5)
displayImg(gradientSvB, 5)


'''
Kernels used for morphology will be of the following forms:

1 0 0 0 0 0 0
0 1 0 0 0 0 0
0 0 1 0 0 0 0
0 0 0 1 0 0 0   x = y
0 0 0 0 1 0 0
0 0 0 0 0 1 0
0 0 0 0 0 0 1

0 0 0 0 0 0 1
0 0 0 0 0 1 0
0 0 0 0 1 0 0
0 0 0 1 0 0 0   -x = y
0 0 1 0 0 0 0
0 1 0 0 0 0 0
1 0 0 0 0 0 0

0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 1 0 0 0   x = 0
0 0 0 1 0 0 0
0 0 0 1 0 0 0
0 0 0 1 0 0 0

0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
1 1 1 1 1 1 1   y = 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0

'''

