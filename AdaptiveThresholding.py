import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler as mms
import os

image = loadGray("DrusenImgs/1.tif")
img = image

testImg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

displayImg(testImg, 20)

# histogram equalization
eqImg = cv2.equalizeHist(img)
displayImg(img, 5)
displayImg(img, 10)

img = eqImg

# identifying pixel ranges of rpe
thresh = 235
img = image

for row in range(0, img.shape[0]):
    for col in range(0, img.shape[1]):
        if img[row][col] >= thresh:
            img[row][col] = 255
        else:
            img[row][col] = 0

displayImg(image, 5)
displayImg(img, 5)


for col in range(0, img.shape[1]):
    for row in range(0, img.shape[0]):
        
        
        
        