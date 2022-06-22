import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local, threshold_yen

import tensorflow as tf
from tensorflow import keras

from config import *


def preprocessor(imgPath, imgSize, binary=False, show=False):
    '''
         This is the part of https://github.com/lamhoangtung/LineHTR with simple modification
         See License.
    '''

    """ Pre-processing image for predicting """
    img = cv2.imread(imgPath)
    # Binary
    if binary:
        brightness = 0
        contrast = 50
        img = np.int16(img)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = threshold_local(img, 11, offset=10, method="gaussian")
        img = (img > T).astype("uint8") * 255

        # Increase line width
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create target image and copy sample image into it
    # img = img * (-1) + 255
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)

    # Scale according to f (result at least 1 and at most wt or ht)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    newSize = (wt, ht)
    # if w > wt:
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_AREA)
    target = np.zeros([ht, wt])
    target[0:newSize[1], 0:newSize[0]] = img

    # Transpose for TF
    img = cv2.transpose(target)
    """img -=128
    img /=128"""

    # Normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    if show:
        plt.imshow(img, cmap='gray')
        plt.show()

    return img


def wer(r, h):
    '''
         This is the part of https://github.com/lamhoangtung/LineHTR with simple modification
         See License.
    '''
    """
    Calculation of WER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    Parameters
    ----------
    r : list
    h : list
    Returns
    -------
    int
    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)]


def str2list(s, maxTextLen):
    t = [wordCharList.index(c) for c in s.split('|') if c in wordCharList]
    return t + [num_classes] * (maxTextLen - len(t))


def truncateLabel(text, maxTextLen):
    # ctc_loss can't compute loss if it cannot find a mapping between text label and input
    # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
    # If a too-long label is provided, ctc_loss returns an infinite gradient
    cost = 0
    l = []
    for i in range(len(text)):
        if i != 0 and text[i] == text[i - 1]:
            cost += 2
            l.append(0)
            l.append(1)
        else:
            cost += 1
            l.append(1)
        if cost > maxTextLen:
            return None
    l = l + [0] * (maxTextLen - cost)
    return l


def ctc_loss_lambda_func(y_true, y_pred):
    """Function for computing the CTC loss"""

    # if len(y_true.shape) > 2:
    # y_true = tf.squeeze(y_true)

    # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
    # output of every model is softmax
    # so sum across alphabet_size_1_hot_encoded give 1
    #               string_length give string length

    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

    # y_true strings are padded with 0
    # so sum of non-zero gives number of characters in this string
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    # average loss across all entries in the batch
    loss = tf.reduce_mean(loss)

    return loss


def map_to_chars(inputs, table, blank_index=0, merge_repeated=False):
    """Map to chars.

    Args:
        inputs: list of char ids.
        table: char map.
        blank_index: the index of blank.
        merge_repeated: True, Only if tf decoder is not used.

    Returns:
        lines: list of string.
    """
    lines = []
    for line in inputs:
        text = ""
        previous_char = -1
        for char_index in line:
            if merge_repeated:
                if char_index == previous_char:
                    continue
            previous_char = char_index
            if char_index == blank_index:
                continue
            # print(char_index, blank_index)
            text += table[char_index]
        lines.append(text)
    return lines


def distance(hypothesis, truth, normalize=True):
    import editdistance
    per = editdistance.eval(hypothesis, truth)
    if normalize:
        per /= len(truth)
    return per


def levenshtein_distance(hypothesis, truth, normalize=True):
    import Levenshtein as lev
    per = lev.distance(hypothesis, truth)
    if normalize:
        per /= len(truth)
    return per


def map_and_count(decoded, Y, mapper, blank_index=0, merge_repeated=False, show=False):
    decoded = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
    # decoded = tf.argmax(decoded).numpy()
    Y = np.stack(list(Y))
    decoded = map_to_chars(decoded, mapper, blank_index=blank_index, merge_repeated=merge_repeated)
    Y = map_to_chars(Y, mapper, blank_index=blank_index, merge_repeated=merge_repeated)
    count_s = 0
    count_w = 0
    count_c = 0
    for y_pred, y in zip(decoded, Y):

        if show:
            print('[{} / {}] SER: {:.2f}, WER: {:.2f}, CER: {:.2f}'.format(y_pred, y, int(y_pred != y),
                                                                           distance(y_pred.split(), y.split(),
                                                                                    True) * 100,
                                                                           distance(y_pred, y, True) * 100))
        if y_pred == y:
            count_s += 1

        count_c += distance(y_pred, y, True)
        count_w += distance(y_pred.split(), y.split(), True)
    return count_s, count_w, count_c
