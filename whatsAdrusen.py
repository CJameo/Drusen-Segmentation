import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler as mms

def loadGray(img):
    image = cv2.imread(img, 0)
    cv2.imshow("Image", image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return image


def loadGray2(img):
    image = cv2.imread(img, 0)
    return image


def displayImg(img, seconds = 5):
    cv2.imshow("Image", img)
    cv2.waitKey(1000 * seconds)
    cv2.destroyAllWindows()
    
    
def minMaxNorm(img, bitrate = 256):
    min = img.min()
    max = img.max()
    newMin = 0
    newMax = bitrate - 1
    newImg = np.zeros(img.shape).astype("uint8")
    
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            oVal = img[row][col]
            nVal = (oVal - min)/(max - min) * (newMax - newMin) + newMin
            newImg[row][col] = nVal
    
    return newImg


def grayLevel(img, levels = 2):

    newImg = np.zeros(img.shape).astype("uint8")    # Zeroes matrix of original dimensions
    tempMax = levels - 1    # For min-max normalization on interval of size scale
    max = img.max() # Current maximum intensity
    min = img.min() # Current minimum intensity

    for row in range(0, newImg.shape[0]):
        for col in range(0, newImg.shape[1]):
            temp = round(((img[row][col] - min) / (max - min)) * (tempMax)) # Quantize on new interval
            newImg[row][col] = round((temp / tempMax) * (max - min) + min)  # Redistribute on original interval

    return newImg


# loading a sample image
image = loadGray("xml data AMD/Drusen 1/1B5E350.tif")
image2 = loadGray("xml data AMD/Drusen 1/E44B37C0.tif")
image3 = loadGray("xml data AMD/Dry 1/9D5A32A0.tif")
image4 = loadGray("xml data AMD/Dry 1/CA0C7560.tif")

# Using simple quantization methods
# Quantization using self implemented method
n = 4
nImage = grayLevel(image, n)

displayImg(image)
displayImg(nImage, 10)
cv2.imwrite("4quantized.jpg", nImage)

n = 4
nImage2 = grayLevel(image2, n)

displayImg(nImage2, 100)
cv2.imwrite("4quantized2.jpg", nImage2)

n = 4
nImage3 = grayLevel(image3, n)

displayImg(image3, 5)
displayImg(nImage3, 5)
cv2.imwrite("4quantized3.jpg", nImage3)

n = 4
nImage4 = grayLevel(image4, n)

displayImg(image4, 5)
displayImg(nImage4, 5)
cv2.imwrite("4quantized4.jpg", nImage4)



# Simple morphological operators on 256 gray levels
rectK = cv2.getStructuringElement(cv2.MORPH_RECT,(11,1))
elipK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
crossK = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

# erosion
testImageER = cv2.erode(image, rectK)
displayImg(image)
displayImg(testImageER)

testImageEE = cv2.erode(image, elipK)
displayImg(image)
displayImg(testImageEE)

# dilation
testImageDR = cv2.erode(image, rectK)
displayImg(image)
displayImg(testImageDR)

# Opening
testImageOR = cv2.morphologyEx(image, cv2.MORPH_OPEN, rectK)
displayImg(image)
displayImg(testImageOR)

testImage = testImageOR
nTestImage = grayLevel(testImage, n)
displayImg(image)
displayImg(testImage)
displayImg(nTestImage)
cv2.imwrite("samplehealthy.jpg", nTestImage)
