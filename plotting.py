"""
File Summary: Handles plotting and displaying images.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

import file_names
import image_reading

def plotImageHistogram(histogram, bar=False, axis=None):
    """
    If bars is True, it's a bar chart, otherwise it's a line chart.
    You will need to call plt.show() outside of this function.
    """
    if axis is None:
        fig, axis = plt.subplots()
    xValues = list(range(0, 256))
    if bar:
        axis.bar(xValues, histogram)
    else:
        axis.plot(xValues, histogram)
    axis.set_title('GrayScale Image Histogram')
    axis.set_ylabel('Frequency')

def plotHistogramWithPeakAndPitLines(histogram, peakIndices, pitIndices, axis=None):
    """
    Peaks are drawn in green and pits are drawn in red.
    """
    if axis is None:
        fig, axis = plt.subplots()

    plotImageHistogram(axis, histogram)

    for i in range(len(peakIndices)):

        peakIndex = peakIndices[i]
        pitIndex = pitIndices[i]

        # If the peak and pit indices are at the same location, we draw one purple line.
        if peakIndex == pitIndex:
            axis.axvline(x=peakIndex, color='purple', linestyle='--', linewidth=0.5)

        # Otherwise, we draw one line for the peak and one for the pit.
        else:
            axis.axvline(x=peakIndex, color='green', linestyle='--', linewidth=0.5)
            axis.axvline(x=pitIndex, color='red', linestyle='--', linewidth=0.5)

    plt.show()

def plotPeakIndices(arrLength, peakIndices, title, axis=None):
    if axis is None:
        fig, axis = plt.subplots()
    x = range(arrLength)
    y = np.zeros(shape=(arrLength))
    y[peakIndices] = 1
    axis.plot(x, y)
    axis.set_ylabel('Peak Indication')
    axis.set_title(title)

def showGrayScaleImage(image, title, axis=None):
    if axis is None:
        fig, axis = plt.subplots()
    axis.imshow(image, cmap='gray')
    axis.set_title(title)
    axis.set_axis_off()

def showColorImage(image, title, axis=None):
    if axis is None:
        fig, axis = plt.subplots()
    image = image_reading.scaleColorImageBetween0And1(image)
    axis.imshow(image)
    axis.set_title(title)
    axis.set_axis_off()

def printFileHasBeenSavedMessage(filePath):
    fileName = os.path.basename(filePath)
    string = 'File {} has been saved'.format(fileName)
    print(string)

def saveAlgorithmStepsPlot(hist, peakList, filePath):
    fig, axes = plt.subplots(5, figsize=(10, 15))
    plotImageHistogram(hist, False, axes[0])
    titleList = ['All Peaks', 'Vertical Peaks', 'Horizontal Peaks', 'Final Peaks (Possibly One Added)']
    axesList = [axes[1], axes[2], axes[3], axes[4]]
    for i in range(len(peakList)):
        plotPeakIndices(len(hist), peakList[i], titleList[i], axesList[i])
    plt.savefig(filePath, dpi=300)
    printFileHasBeenSavedMessage(filePath)

def saveOldNewImageComparisonPlot(originalIm, segmentedIm, filePath):
    # Plot the original and new images.
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    originalTitle, segmentedTitle = 'Original Image', 'Segmented Image'

    if image_reading.imageIsGrayScale(originalIm):
        showGrayScaleImage(originalIm, originalTitle, axes[0])
        showGrayScaleImage(segmentedIm, segmentedTitle, axes[1])
    else:
        showColorImage(originalIm, originalTitle, axes[0])
        showColorImage(segmentedIm, segmentedTitle, axes[1])

    plt.savefig(filePath, dpi=300)
    printFileHasBeenSavedMessage(filePath)


if __name__ == '__main__':
    files = file_names.getColorImageFilePaths()
    image = image_reading.readImage(files[0])
    showColorImage(image, 'Image')
    plt.show()

