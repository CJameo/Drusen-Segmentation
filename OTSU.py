import numpy as np
from numpy import polynomial as poly
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
    

image = cv2.imread("DryImgs/2.tif", 0)
displayImg(image, 5)

img = image

# OTSU


# Gaussian blur + Otsu
blur = cv2.GaussianBlur(img,(5,5),0)
cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

displayImg(th3, 10)
cv2.imwrite("samples/dry2otsu.jpg", th3)


# applying median filter
med = cv2.medianBlur(th3, 5)
displayImg(med, 5)
cv2.imwrite("samples/dry2otsumed.jpg", med)

# morphological kernel initialization
rectK = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17)) #MORPH_X DESCRIBES SHAPE TUPLE SPECIFIES KERNEL DIMENSIONS
elipK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
crossK = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

# morphological closing
closed = cv2.morphologyEx(med, cv2.MORPH_CLOSE, rectK)
displayImg(closed, 5)

cv2.imwrite("samples/dry2closed.jpg", closed)



# Generating Images
image = cv2.imread("DryImgs/4.tif", 0)
displayImg(image, 5)

img = image

blur = cv2.GaussianBlur(img,(5,5),0)
cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

med = cv2.medianBlur(th3, 5)
displayImg(med, 5)

rectK = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17)) #MORPH_X DESCRIBES SHAPE TUPLE SPECIFIES KERNEL DIMENSIONS
closed = cv2.morphologyEx(med, cv2.MORPH_CLOSE, rectK)
displayImg(closed, 5)

cv2.imwrite("samples/dry4closed.jpg", closed)


# fit polynomial
x = img[0][:]
y = img[:][0]
x2 = [0, :]
y2 = 
test = poly.Polynomial.fit(x, y, 2)

test

# open cv fit line
test = cv2.fitLine(image, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)