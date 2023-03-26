"""
File Summary: Supports determining the name of files that will be saved.
"""

import os

GENERATED_IMAGES_DIRECTORY = 'Generated Images'

def getColorImageFilePaths():
    directory = 'Color Images'
    return [os.path.join(directory, fileName) for fileName in os.listdir(directory)]

def getGrayScaleImageFilePaths():
    directory = r'C:\Program Files\MATLAB\R2021a\toolbox\images\imdata'
    fileNames = ['AT3_1m4_01.tif', 'bag.png', 'cameraman.tif', 'coins.png', 'glass.png', 'moon.tif', 'mri.tif', 'pout.tif']
    return [os.path.join(directory, fileName) for fileName in fileNames]

def getGeneratedImagesFilePaths():
    return [os.path.join(GENERATED_IMAGES_DIRECTORY, fileName) for fileName in os.listdir(GENERATED_IMAGES_DIRECTORY)]

def removeExtensionFromFileName(fileName):
    index = fileName.find('.')
    return fileName[:index]

def getPeaksImageFilePath(origFilePath, channelColor=None):
    """
    A channelColor can be included here if you are retrieving peaks for a color image.
    """
    fileName = os.path.basename(origFilePath)
    fileName = removeExtensionFromFileName(fileName)

    if channelColor is not None:
        fileName += ' ' + channelColor

    fileName = fileName.title() + ' Peaks Image.png'
    return os.path.join(GENERATED_IMAGES_DIRECTORY, fileName)

def getSegmentedImageFilePath(origFilePath):
    fileName = os.path.basename(origFilePath)
    fileName = removeExtensionFromFileName(fileName)
    fileName = fileName.title() + ' Segmented Image.png'
    return os.path.join(GENERATED_IMAGES_DIRECTORY, fileName)


if __name__ == '__main__':
    print(getColorImageFilePaths())

