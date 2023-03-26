"""
File Summary: Implements the algorithm according to my interpretation.
"""

import numpy as np
from sklearn.cluster import KMeans
import random

import plotting
import file_names
import image_reading


def getImageHistogram(image, normalize):
    histogram = np.zeros(shape=(256,))
    for row in image:
        for val in row:
            histogram[val] += 1
    if normalize:
        histogram = histogram / histogram.sum()
    return histogram

def findPeakAndPitIndicesWithEnforcedOrder(array):
    """
    This version of the function enforces the peak - pit order. Don't allow two peaks or two pits to come in a row.
    It's the only way I've gotten the algorithm to work on all of the images.
    """

    # Iterate over the histogram. Only allow a peak after a pit and a pit after a peak.
    peakIndices = []
    pitIndices = []
    peakActive = True
    pitActive = True
    for i in range(1, len(array) - 1):
        if pitActive and array[i-1] < array[i] > array[i+1]:
            peakIndices.append(i)
            peakActive = True
            pitActive = False

        if peakActive and array[i-1] > array[i] < array[i+1]:
            pitIndices.append(i)
            pitActive = True
            peakActive = False

    # At the end of the iteration, there could be one fewer peak or one fewer pit, which must be corrected.
    # If a peak is missing, then we just duplicate the previous peak.
    # If a pit is missing, then we just duplicate the previous pit.
    if len(peakIndices) < len(pitIndices):
        peakIndices.append(peakIndices[-1])
    elif len(peakIndices) > len(pitIndices):
        pitIndices.append(pitIndices[-1])

    return np.array(peakIndices), np.array(pitIndices)

def filterByVerticalDistance(hist, peakIndices, pitIndices):
    """
    We keep peak indices that have a peak - pit distance greater than the average.
    """
    assert len(peakIndices) == len(pitIndices), 'The number of peaks is {} and the number of pits is {}.'.format(len(peakIndices), len(pitIndices))
    verticalDistArray = np.abs(hist[peakIndices] - hist[pitIndices])
    avgVerticalDist = verticalDistArray.mean()
    return peakIndices[verticalDistArray > avgVerticalDist]

def getBidirectionalArrayDifference(array):
    """
    Get the difference of each array element from its two neighbors.
    """
    diffList = []
    for i in range(len(array)):

        # If the current index is 0, we can only look to the right.
        if i == 0:
            meanDiff = abs(array[i] - array[i + 1])

        # If the current index is the last element, we can only look to the left.
        elif i == (len(array) - 1):
            meanDiff = abs(array[i] - array[i - 1])

        # Otherwise, we can look in both directions.
        else:
            diffRight = abs(array[i] - array[i + 1])
            diffLeft = abs(array[i] - array[i - 1])
            meanDiff = (diffLeft + diffRight) / 2

        diffList.append(meanDiff)

    return np.array(diffList)

def filterByBidirectionalHorizontalDistance(hist, verticalPeakIndices):
    diffArray = getBidirectionalArrayDifference(verticalPeakIndices)
    # diffArray = diffArray * hist[verticalPeakIndices]  # This line is based on the paper but seems to worsen performance.
    avgHorizontalDiff = np.mean(diffArray)
    return verticalPeakIndices[diffArray >= avgHorizontalDiff]

def addOneRandomCentroid(currCentroid):
    """
    This function should be called if you end up with only one centroid. You need at least two.
    It selects a random number that isn't equal to the current centroid.
    """
    while True:
        chosenVal = random.randint(0, 255)
        if chosenVal != currCentroid:
            return chosenVal

def addFarthestCentroid(currCentroid):
    if currCentroid > 127:
        return 5
    else:
        return 250

def findInitialCentroidsForChannel(imageChannel, peakImFileName=None):
    """
    In the case of an RGB image, this function would need to be called three times.
    If peakImFileName is not None, we save an image summarizing the algorithm steps.
    """
    hist = getImageHistogram(imageChannel, False)
    peakIndices, pitIndices = findPeakAndPitIndicesWithEnforcedOrder(hist)

    verticalPeakIndices = filterByVerticalDistance(hist, peakIndices, pitIndices)
    horizontalPeakIndices = filterByBidirectionalHorizontalDistance(hist, verticalPeakIndices)

    # Add a new centroid if we only have one.
    finalPeaks = list(horizontalPeakIndices)
    if len(finalPeaks) == 1:
        newCentroid = addFarthestCentroid(finalPeaks[0])
        finalPeaks.append(newCentroid)

    peakList = [peakIndices, verticalPeakIndices, horizontalPeakIndices, finalPeaks]

    if peakImFileName is not None:
        plotting.saveAlgorithmStepsPlot(hist, peakList, peakImFileName)

    return finalPeaks

def getKMeansPredictions(centroids, imageArray):
    """
    Flatten the data, run the K means algorithm, and reshape the data to its correct form.
    An image of clusters is returned.
    """
    centroids = np.array(centroids).reshape(-1, 1)  # Modification so sklearn works.
    kMeans = KMeans(n_clusters=len(centroids), init=centroids, n_init=1, max_iter=1000)
    flattenedImage = imageArray.flatten().reshape(-1, 1)
    pred = kMeans.fit_predict(flattenedImage)
    return pred.reshape(imageArray.shape)

def runAlgorithmOnSingleImage(filePath, saveStepsImage):
    """
    Function can run on both color and grayscale images.
    """
    originalIm = image_reading.readImage(filePath)

    # The case for grayscale images.
    if image_reading.imageIsGrayScale(originalIm):
        if saveStepsImage:
            algorithmStepsFileName = file_names.getPeaksImageFilePath(filePath)
            centroids = findInitialCentroidsForChannel(originalIm, algorithmStepsFileName)
        else:
            centroids = findInitialCentroidsForChannel(originalIm)
        segmentedIm = getKMeansPredictions(centroids, originalIm)

    # The case for color images.
    else:
        segmentedIm = np.empty(originalIm.shape)
        colorChannels = ['Red', 'Green', 'Blue']
        for i in range(3):
            if saveStepsImage:
                algorithmStepsFileName = file_names.getPeaksImageFilePath(filePath, colorChannels[i])
                centroids = findInitialCentroidsForChannel(originalIm[:, :, i], algorithmStepsFileName)
            else:
                centroids = findInitialCentroidsForChannel(originalIm[:, :, i])
            segmentedChannel = getKMeansPredictions(centroids, originalIm[:, :, i])
            segmentedIm[:, :, i] = segmentedChannel

    # Run k means and save the actual and segmented images side by side.
    segmentedImFilePath = file_names.getSegmentedImageFilePath(filePath)
    plotting.saveOldNewImageComparisonPlot(originalIm, segmentedIm, segmentedImFilePath)


def runOnAllGrayScaleImages(saveStepsImage):
    fileNames = file_names.getGrayScaleImageFilePaths()
    for fileName in fileNames:
        try:
            runAlgorithmOnSingleImage(fileName, saveStepsImage)
        except Exception as e:
            print('Exception:', e)

def runOnAllColorImages(saveStepsImage):
    fileNames = file_names.getColorImageFilePaths()
    for fileName in fileNames:
        try:
            runAlgorithmOnSingleImage(fileName, saveStepsImage)
        except Exception as e:
            print('Exception:', e)


if __name__ == '__main__':
    runOnAllGrayScaleImages(True)
    runOnAllColorImages(True)

