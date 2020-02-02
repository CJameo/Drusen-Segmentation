import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler as mms
import os


LBPImage = test[1]['img']
displayImg(LBPImage)
oImage = test[0]['img']
displayImg(oImage)
testLBPImage = np.zeros(oImage.shape).astype("uint8")

for row in range(0, oImage.shape[0]):
    for col in range(0, oImage.shape[1]):
        val = oImage[row][col] - LBPImage[row][col]
        testLBPImage[row][col] = max(val, 0)



displayImg(testLBPImage)


# testing vertical gradient
# gaussian
img = image
blurImg = cv2.GaussianBlur(img, (5, 5), 0)
displayImg(blurImg)

'''
blurImg = cv2.GaussianBlur(img, (5, 5), 1)
displayImg(blurImg)
blurImg = cv2.GaussianBlur(img, (5, 5), 1.5)
displayImg(blurImg)
blurImg = cv2.GaussianBlur(img, (5, 5), 2)
displayImg(blurImg)
'''

# sobel y
sobely = cv2.Sobel(blurImg,cv2.CV_64F, dx=0, dy=1, ksize=5)
displayImg(-sobely, 10)


# difference of gaussians
img = image
blurImg = cv2.GaussianBlur(img, (5, 5), 1.5)



# LoG estimation
img = image
oct10 = img
displayImg(oct10, 3)
oct11 = cv2.GaussianBlur(oct10, ksize = (5, 5), sigmaX=0)
displayImg(oct11, 3)
oct12 = cv2.GaussianBlur(oct11, ksize = (5, 5), sigmaX=0)
displayImg(oct12, 3)
oct13 = cv2.GaussianBlur(oct12, ksize = (5, 5), sigmaX=0)
displayImg(oct13, 3)
oct14 = cv2.GaussianBlur(oct13, ksize = (5, 5), sigmaX=0)
displayImg(oct14, 3)
oct15 = cv2.GaussianBlur(oct14, ksize = (5, 5), sigmaX=0)
displayImg(oct15, 3)

oct20 = cv2.resize(img, dsize=(oct10.shape[0]//2, oct10.shape[1]//2))
displayImg(oct20, 3)
oct21 = cv2.GaussianBlur(oct20, ksize = (5, 5), sigmaX=0)
displayImg(oct21, 3)
oct22 = cv2.GaussianBlur(oct21, ksize = (5, 5), sigmaX=0)
displayImg(oct22, 3)
oct23 = cv2.GaussianBlur(oct22, ksize = (5, 5), sigmaX=0)
displayImg(oct23, 3)
oct24 = cv2.GaussianBlur(oct23, ksize = (5, 5), sigmaX=0)
displayImg(oct24, 3)
oct25 = cv2.GaussianBlur(oct24, ksize = (5, 5), sigmaX=0)
displayImg(oct25, 3)

oct30 = cv2.resize(img, dsize=(oct20.shape[0]//2, oct20.shape[1]//2))
displayImg(oct30, 3)
oct31 = cv2.GaussianBlur(oct30, ksize = (5, 5), sigmaX=0)
displayImg(oct31, 3)
oct32 = cv2.GaussianBlur(oct31, ksize = (5, 5), sigmaX=0)
displayImg(oct32, 3)
oct33 = cv2.GaussianBlur(oct32, ksize = (5, 5), sigmaX=0)
displayImg(oct33, 3)
oct34 = cv2.GaussianBlur(oct33, ksize = (5, 5), sigmaX=0)
displayImg(oct34, 3)
oct35 = cv2.GaussianBlur(oct34, ksize = (5, 5), sigmaX=0)
displayImg(oct35, 3)

cv2.imwrite("samples/oct10.jpg", oct10)
cv2.imwrite("samples/oct101.jpg", (oct10-oct11))
cv2.imwrite("samples/oct112.jpg", (oct11-oct12))
cv2.imwrite("samples/oct123.jpg", (oct12-oct13))
cv2.imwrite("samples/oct134.jpg", (oct13-oct14))
cv2.imwrite("samples/oct145.jpg", (oct14-oct15))

cv2.imwrite("samples/oct30.jpg", oct30)
cv2.imwrite("samples/oct301.jpg", (oct30-oct31))
cv2.imwrite("samples/oct312.jpg", (oct31-oct32))
cv2.imwrite("samples/oct323.jpg", (oct32-oct33))
cv2.imwrite("samples/oct334.jpg", (oct33-oct34))
cv2.imwrite("samples/oct345.jpg", (oct34-oct35))

