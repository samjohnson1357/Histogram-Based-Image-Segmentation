"""
File Summary: Implements reading and scaling of images.
"""

import matplotlib.image as mpimg
import numpy as np

def rescaleImage(im):
    return (im * 255).astype(np.uint8)

def readImage(filePath):
    """
    Matplotlib returns the image as either np.uint8 from 0 to 255 or np.float32 from 0 to 1.
    We must return each image in the uint8 format.
    """
    im = mpimg.imread(filePath)
    if im.dtype == np.float32:
        return rescaleImage(im)
    return im

def imageIsGrayScale(im):
    return im.ndim == 2

def scaleColorImageBetween0And1(image):
    newImage = np.empty(image.shape)
    for i in range(3):
        channel = image[:, :, i]
        maxVal = np.max(channel)
        minVal = np.min(channel)
        newChannel = (channel - minVal) / (maxVal - minVal)
        newImage[:, :, i] = newChannel
    return newImage

