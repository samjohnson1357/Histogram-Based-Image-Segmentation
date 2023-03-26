"""
File Summary: It helps us create LaTeX figures quickly and easily. This will make writing the paper much easier.
"""

import textwrap
import file_names

def getFigureText(width, filePath, caption, label):
    string = r"""
    \begin{{figure}}[h]
        \centering
        \includegraphics[width={}\textwidth]{{{}}}
        \caption{{{}}}
        \label{{{}}}
    \end{{figure}}
    """.format(width, filePath, caption, label)
    return textwrap.dedent(string)

def saveTextToFile(string, fileName):
    file = open(fileName, 'wt')
    file.write(string)
    file.close()

def createFigureTextForAllGeneratedImages(fileName):
    figureString = ''
    for filePath in file_names.getGeneratedImagesFilePaths():
        filePath = swapSlashes(filePath)
        figureString += getFigureText(0.5, filePath, filePath, filePath)
    saveTextToFile(figureString, fileName)

def swapSlashes(string):
    import os
    newPath = string.replace(os.sep, '/')
    return newPath


def isPeaksImage(filePath):
    return 'Peaks' in filePath

def determineCaption(filePath):
    if isPeaksImage(filePath):
        return


if __name__ == '__main__':
    createFigureTextForAllGeneratedImages('Color Images.txt')

