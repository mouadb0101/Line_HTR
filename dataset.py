from utils import preprocessor
import re, os
from config import *


def readIAMData(filePath, imgSize):
    '''
           This is the part of https://github.com/githubharald/SimpleHTR with simple modification
           See License.
    '''

    data = {}

    f = open(filePath, 'r')
    for line in f:

        # ignore comment line
        if not line or line[0] == '#':
            continue

        lineSplit = line.strip().split(' ')
        assert len(lineSplit) >= 9

        fileName = 'lines/' + lineSplit[0] + '.jpg'

        gtText = ' '.join(lineSplit[8:]).replace('|', ' ')

        # gtText = gtText.lower()
        gtText = re.sub(r'([\'?.!,¿])', r" \1 ", gtText)
        gtText = re.sub(r'[^ ' + wordCharList + ']+', ' ', gtText)
        gtText = re.sub(r'[" "]+', ' ', gtText)

        lineSplit = list(gtText)
        gtText = '|'.join(lineSplit[:])

        gtText = gtText.rstrip().strip()

        # if truncateLabel(gtText, maxTextLen) :
        # print("maxlen must be changed")
        # print(gtText)
        # chars = chars.union(set(list(gtText)))

        img = preprocessor(fileName, imgSize, False, False)
        data[gtText] = img

    return data


def readKHATTdata(filePath, imgSize):
    data = {}

    f = open(filePath, 'r')
    for line in f:

        # ignore comment line
        if not line or line[0] == '#':
            continue

        lineSplit = line.strip().split('	 ')
        assert len(lineSplit) == 2

        fileNameSplit = lineSplit[0].split('.')
        fileName = 'lines/' + filePath.split('.')[0] + '/' + fileNameSplit[0] + '.jpg'

        lineSplit = lineSplit[1].split(" ")
        gtText = '|'.join(lineSplit[:])

        gtText = gtText.rstrip().strip()

        # if truncateLabel(gtText, maxTextLen) :
        # print("maxlen must be changed")
        # print(gtText)
        # chars = chars.union(set(list(gtText)))
        if os.path.isfile(fileName):
            img = preprocessor(fileName, imgSize, False, False)
            data[gtText] = img
        else:
            # print(fileName)
            pass
    return data


def readParzival_SaintGallData(filePath, imgSize):
    data = {}

    f = open(filePath, 'r')
    for line in f:

        # ignore comment line
        if not line or line[0] == '#':
            continue

        lineSplit = line.strip().split(' ')
        assert len(lineSplit) >= 2

        fileName = 'lines/' + lineSplit[0] + '.jpg'

        gtText = '|'.join(lineSplit[1:])

        gtText = gtText.rstrip().strip()

        # if truncateLabel(gtText, maxTextLen) :
        # print("maxlen must be changed")
        # print(gtText)
        # chars = chars.union(set(list(gtText)))

        if os.path.isfile(fileName):
            img = preprocessor(fileName, imgSize, False, False)
            data[gtText] = img
        else:
            # print(fileName)
            pass
    return data


if __name__ == "__main__":
    data = readIAMData(train_path, (img_w, img_h))
    print(len(data))
    dataVal = readIAMData(validation_path, (img_w, img_h))
    print(len(dataVal))
    dataTest = readIAMData(test_path, (img_w, img_h))
    print(len(dataTest))
