"""
File Summary: Some alternate approaches that I implemented but ultimately decided against using.
This is all part of the development process. You can skip over this file if you like.
"""

import numpy as np
import pandas as pd
import cv2
import os

def readGrayScaleImage(fileName):
    IMAGE_DIRECTORY = r'C:\Program Files\MATLAB\R2021a\toolbox\images\imdata'
    path = os.path.join(IMAGE_DIRECTORY, fileName)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def getShiftedHistogramDf(histogram):
    """
    Function prepares a dataframe for further use with finding peak and pit points.
    """
    df = pd.DataFrame(data=histogram, index=range(0, 256), columns=['Histogram'])
    df.index.name = 'Intensity'
    df['Previous'] = df['Histogram'].shift(1)
    df['Next'] = df['Histogram'].shift(-1)
    df = df[['Previous', 'Histogram', 'Next']]
    df.dropna(inplace=True)
    return df

def findPeakPointIndices(df):
    """
    A peak occurs when there is an increase and then a decrease.
    """
    peakFilter = (df['Histogram'] > df['Previous']) & (df['Next'] < df['Histogram'])
    return df[peakFilter].index.values

def findPitPointIndices(df):
    """
    A pit occurs when there is a decrease and then an increase.
    """
    pitFilter = (df['Histogram'] < df['Previous']) & (df['Next'] > df['Histogram'])
    return df[pitFilter].index.values

def findPeakAndPitPointIndices(histogram):
    df = getShiftedHistogramDf(histogram)
    peakPoints = findPeakPointIndices(df)
    pitPoints = findPitPointIndices(df)
    return peakPoints, pitPoints

def alternateGetImageHistogram(image):
    return np.histogram(image, range(0, 256))

def getPeakIndicesAboveAverageChange(peakIndices, pitIndices):
    diffArray = np.abs(pitIndices - peakIndices)
    meanDiff = diffArray.mean()
    return peakIndices[diffArray > meanDiff]

def peaksAndPitsComeInCorrectOrder(peakIndices, pitIndices):
    """
    The order of peak - pit - peak must be maintained. Basically, this means that subtracting one from the other
    should yield either all positive or all negative values.
    """
    boolArray = (peakIndices - pitIndices) > 0
    return all(boolArray) or all(~boolArray)

def alternateFindPeakAndPitIndicesWithLimits(array):
    """
    This version of the function enforces the peak - pit order.
    """
    peakIndices = []
    pitIndices = []
    peakActive = True
    pitActive = True
    for i in range(1, len(array)):
        if pitActive and array[i-1] < array[i] > array[i+1]:
            peakIndices.append(i)
            peakActive = True
            pitActive = False

        if peakActive and array[i-1] > array[i] < array[i+1]:
            pitIndices.append(i)
            pitActive = True
            peakActive = False

    return np.array(peakIndices), np.array(pitIndices)

def showImage(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def modifyColorImageCentroids(centroidList):
    """
    There are three channels for color images. That means we must adjust cluster centroids so each channel has
    the same number.
    """
    # Find the minimum length.
    lengths = []
    for centroid in centroidList:
        lengths.append(len(centroid))
    minLength = np.min(lengths)

    # Create the new centroids.
    newCentroidList = []
    for centroid in centroidList:
        newCentroidList.append(centroid[0:minLength])

    return newCentroidList

def findPeakAndPitIndices(array):
    """
    Passed over in favor of the enforced order function.
    """
    peakIndices = []
    pitIndices = []
    for i in range(1, len(array) - 1):

        # A peak is defined as an increase and then a decrease.
        if array[i-1] < array[i] > array[i+1]:
            peakIndices.append(i)

        # A pit is defined as a decrease and then an increase.
        if array[i-1] > array[i] < array[i+1]:
            pitIndices.append(i)

    return np.array(peakIndices), np.array(pitIndices)

def getShiftedArrayDifference(array):
    """
    Passed over in favor of the bidirectional approach.
    """
    shiftedDiffList = []
    for i in range(len(array) - 1):
        diff = abs(array[i] - array[i + 1])
        shiftedDiffList.append(diff)
    return shiftedDiffList

def filterByHorizontalDistance(verticalPeakIndices):
    diffArray = getShiftedArrayDifference(verticalPeakIndices)
    diffArray.append(diffArray[-1])  # Last value is appended because it is skipped when shifting.
    avgHorizontalDiff = np.mean(diffArray)
    return verticalPeakIndices[diffArray > avgHorizontalDiff]

