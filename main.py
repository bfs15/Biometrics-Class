#! python2

from multiprocessing import Pool
import load
# import enhance
# import fingerprint
# import compare
# import stats

from PIL import Image
import numpy as np
import math
from matplotlib import pylab as plt
import scipy
import cv2
import sys
import matplotlib.cm
import time
import itertools
import random

from numba import jit

from skimage.feature import hog
import sklearn
from skimage import data, exposure
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection

verbose = True
classPos = 1
classNeg = 0
NMSThresh = 0.5


def sobel_filter(img, axis):
    kernel = [
        np.array([[1, 0, -1]], dtype=np.float),
        np.array([[1], [0], [-1]], dtype=np.float)
    ]

    convolved = scipy.signal.convolve2d(
                img, kernel[axis], mode='same', boundary='symm')

    return convolved


def gradient(img):
    """
    return mag, angle
    """
    #
    gx = sobel_filter(img.astype(np.float), 0)
    gy = sobel_filter(img.astype(np.float), 1)
    #
    # gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    # gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    #
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    return mag, angle


def pyramidCreate(image, levelsUp=8, levelsDown=6, scale=0.05):
    """
    image : image matrix
    levels : quantity of levels in the pyramid
    returns [imageOriginal, imageLevel1, imageLevel2, ...]
    """
    pyr = []
    pyr.append(image)

    levels = [range(1, levelsUp+1), range(-levelsDown, 0)]

    for level in itertools.chain(*levels):
        newScale = (1+scale*level)
        newImg = cv2.resize(image, None, fx=newScale, fy=newScale)
        pyr.append(newImg)
        # print(level, newImg.shape)

    return pyr


@jit(nopython=True)
def GradientHistogram(mag, angle, cellSz=8, binSz=9):
    binsShape = (mag.shape[0]//cellSz, mag.shape[1]//cellSz, binSz)
    bins = np.empty(binsShape)

    angle = (angle - 90) % 180
    for bins_y in range(bins.shape[0]):
        for bins_x in range(bins.shape[1]):
            cellMag = mag[(bins_y+0)*cellSz: (bins_y+1)*cellSz,
                          (bins_x+0)*cellSz: (bins_x+1)*cellSz].flatten()
            cellAngle = angle[(bins_y+0)*cellSz: (bins_y+1)*cellSz,
                              (bins_x+0)*cellSz: (bins_x+1)*cellSz].flatten()

            # tri-linear histogram
            hist = np.zeros((binSz))
            for angleEl, magEl in zip(cellAngle, cellMag):
                pos = angleEl/(180/binSz) - 0.5
                index = int(math.floor(pos)) % binSz
                weight = abs(pos - int(pos))
                hist[index] = (1-weight)*magEl
                hist[(index+1) % binSz] = (weight)*magEl

            # hist, bin_edges = np.histogram(cellAngle, bins=binSz, range=(0, 180), normed=None, weights=cellMag, density=None)
            
            bins[bins_y, bins_x] = hist
    
    return bins

# TODO: @jit(nopython=True)
def normalizeHistogram(bins, blkSz=2, stride=1, eps=1e-6):
    """
    bins : shape (imgHeight//cellSz, imgHeight//cellSz, histogramSize)
    blkSz : block size in cells, a block will be a square of (blkSz x blkSz) cells
    stride : stride in cells, ex: blkSz=2 stride=1 has an overlap of 50%
    """
    binsNormShape = ((bins.shape[0] - blkSz+1)//stride, (bins.shape[1] - blkSz+1)//stride, bins.shape[-1]*(blkSz**2))
    binsNorm = np.empty(binsNormShape)

    for bins_y in range(binsNorm.shape[0]):
        for bins_x in range(binsNorm.shape[1]):
            block = bins[(bins_y*stride): (bins_y*stride) + blkSz,
                                     (bins_x*stride): (bins_x*stride) + blkSz]
            
            blockFlat = block.flatten()
            L2Norm = np.linalg.norm(blockFlat) + eps
            binsNorm[bins_y, bins_x] = blockFlat / L2Norm
            # TODO: remove this?
            if(np.isnan(binsNorm[bins_y, bins_x]).any()):
                binsNorm[bins_y, bins_x] = np.zeros(binsNormShape)

    return binsNorm


def windowHog(image):
    """
    image : image window
    return feature vector of the window
    """
    mag, angle = gradient(image)
    bins = GradientHistogram(mag, angle)
    binsNorm = normalizeHistogram(bins)

    # fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', visualise=True)

    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    # cv2.imshow("hog_image"+str(image_arg), hog_image_rescaled)

    # winSize = (64, 128)
    # blockSize = (16, 16)
    # blockStride = (8, 8)
    # cellSize = (8, 8)
    # nbins = 9
    # hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # winStride = (8, 8)
    # padding = (8, 8)
    # hist = hog.compute(image, winStride, padding)
    return binsNorm.flatten()


def extractWindows(image, stride=8, winSz=(128, 64)):
    """
    yield (windowNdarray, (startY, startX))
    """
    borderY = winSz[0]//4
    borderX = winSz[1]//4
    image = cv2.copyMakeBorder(
        image, borderY, borderY, borderX, borderX, cv2.BORDER_REPLICATE)
    for winBegY in range(0, image.shape[0]-winSz[0], stride):
        for winBegX in range(0, image.shape[1]-winSz[1], stride):
            winEndY = winBegY + winSz[0]
            winEndX = winBegX + winSz[1]
            yield (image[winBegY:,  winBegX: winEndX],  np.array([winBegY, winBegX, winEndY, winEndX]))


def extractWindowsRandom(image, windowNo=2, winSz=(128, 64)):
    image = cv2.copyMakeBorder(
        image, winSz[0]//4, winSz[0]//4, winSz[1]//4, winSz[1]//4, cv2.BORDER_REPLICATE)
    
    start_y = random.randint(0, winSz[0]//4)
    start_x = random.randint(0, winSz[1]//4)
    all_y = range(start_y, image.shape[0]-winSz[0], winSz[0])
    all_x = range(start_x, image.shape[1]-winSz[1], winSz[1])

    windowNo = min(windowNo, len(all_y), len(all_x))

    while True:
        try:
            wins_y = random.sample(all_y, windowNo)
            wins_x = random.sample(all_x, windowNo)
            break
        except ValueError as err:
            if str(err) == "sample larger than population":
                # decrease sample
                print("sample larger than population")
                windowNo -= 1
    
    for win_y, win_x in zip(wins_y, wins_x):
        yield image[win_y: win_y + winSz[0],  win_x: win_x + winSz[1]]


def extractFeaturesNeg(imagesNeg):
    x_train = []
    y_train = []
    for image_arg, image in enumerate(imagesNeg):
        # skip some images for faster testing
        if not image_arg % 12 == 0:
            continue
        #
        image = np.array(image)
        # for imageLevel in pyramidCreate(image, 4, 3, 0.1):
        for imageLevel in pyramidCreate(image):
            for window in extractWindowsRandom(imageLevel):
                # show extracted window
                # cv2.imshow("win", window)
                # cv2.waitKey(64)
                features = windowHog(window)

                x_train.append(features)
                y_train.append(classNeg)  # negative class
                if(np.isnan(features).any()):
                    print("- warning: nan features")
                    print(window.shape, features)
                    sys.stdout.flush()

    return x_train, y_train


def extractFeaturesPos(imagesPos):
    x_train = []
    y_train = []
    for image_arg, image in enumerate(imagesPos):
        image = np.array(image)

        features = windowHog(image)

        x_train.append(features)
        y_train.append(classPos)  # positive class

        # cv2.imshow("image"+str(image_arg), image)
        # cv2.waitKey(64)
        # cv2.destroyAllWindows()
        
    return x_train, y_train


# TODO: @jit(nopython=True)
def nonMaxSuppresion(boxes, probas, overlapThresh):
    """
    boxes: 2darray of shape (n, 4) / n is the number of boxes
    probas: 1darray of shape (n), each element is the probability of the corresponding box
    overlapThresh: if boxes exeed this threshold of overlap with the maximum box, it is suppressed
    """
    if len(boxes) == 0:
        # no boxes, return an empty list
        return []
    # if the bounding boxes are integers
    # convert them to floats, float division is faster
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # list of max boxes indexes
    maxBoxIdxs = []
    # coordinates of the bounding boxes
    begY = boxes[:, 0]
    begX = boxes[:, 1]
    endY = boxes[:, 2]
    endX = boxes[:, 3]
    # area of the bounding boxes
    area = (endY - begY + 1) * (endX - begX + 1)
    # sort boxes by the probabilities
    argProbas = np.argsort(probas)
    # loop removing boxes from the idx list, until no boxes remain
    while len(argProbas) > 0:
        # get the max proba idx in the idxs list (last in a sorted array)
        idxMax = argProbas[-1]
        # save max box idx in the list (to return them later)
        maxBoxIdxs.append(idxMax)
        # calculate overlap:
        # get bot-rightmost beginning (x, y) coordinates of both the boxes
        # and the top-leftmost ending (x, y) coordinates of both the boxes
        # the area of overlap is the area of the box of those coordinates
        # use np.maximum to calculate overlap for every box, is the same as:
        # max(begY[idxMax] - begY[idx]) for idx in argProbas[:-1] (every box except the max one)
        overlBegX = np.maximum(begY[idxMax], begY[argProbas[:-1]])
        overlBegY = np.maximum(begX[idxMax], begX[argProbas[:-1]])
        overlEndX = np.minimum(endY[idxMax], endY[argProbas[:-1]])
        overlEndY = np.minimum(endX[idxMax], endX[argProbas[:-1]])
        # width and height of the overlap box
        # the normal calculation (end-beg+1) can be negative in the cases boxes don't overlap
        overlH = np.maximum(0, overlEndY - overlBegY + 1)
        overlW = np.maximum(0, overlEndX - overlBegX + 1)
        # overlap ratio
        overlArea = overlH * overlW
        overlRatio = (overlArea) / area[argProbas[:-1]]
        # box idxs which overlap with the max box is over the threshold
        overlBoxIdxs = np.where(overlRatio > overlapThresh)[0]
        # delete boxes that overlap and the max proba box
        idxsDelete = np.concatenate((overlBoxIdxs, [-1]))
        argProbas = np.delete(argProbas, idxsDelete)

    # return boxes from max box idxs
    return boxes[maxBoxIdxs]

def predictImage(clf, img):
    # array of probability of predictions for windows
    peopleProb = []
    # array of the position of windows
    peopleWin = []
    winSz = (128, 64)
    lvlsUp = 8
    lvlsDown = 6
    scale = 0.05
    lvl = 2
    for argLvl, imgLvl in enumerate(pyramidCreate(img)):
        if argLvl > lvlsUp:
            # negative levels, down in the pyramid
            argLvl = argLvl-(lvlsUp+lvlsDown+1)

        # if scale decreases, window increases in relative size
        # e.g. the window in the 0.5 scaled image has double the size (1/0.5)
        winScale = 1/(1+scale*argLvl)
        for win, winBoxLvl in extractWindows(imgLvl):
            winFeats = windowHog(win)
            pred = clf.predict_proba(winFeats)
            if pred[classNeg] > pred[classPos]:
                # negative
                continue
            peopleProb.append(pred[classPos])
            # from the box in this pyrLvl, get the real widow
            winBox = winBoxLvl * winScale
            peopleWin.append(winBox)
    
    # non maximum suppresion
    peopleBoxes = nonMaxSuppresion(peopleWin, peopleProb, NMSThresh)

    return peopleBoxes
        

if __name__ == "__main__":
    total_time = 0

    imagesPos = load.database(
        "/home/html/inf/menotti/ci1028-191/INRIAPerson/70X134H96/Test/pos")
    imagesNeg = load.database("/home/html/inf/menotti/ci1028-191/INRIAPerson/Train/neg/")

    start_time = time.time()


    # processorN = 4


    # def splitListN(a, n):
    #     k, m = divmod(len(a), n)
    #     return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
        
    # # list of lists of negative imgs
    # imagesNegChunks = splitListN(list(imagesNeg), processorN)

    # p = Pool(processorN)
    # poolResult = p.map(extractFeaturesNeg, imagesNegChunks)
    
    # x_trainNeg, y_trainNeg = [], []

    # for result in poolResult:
    #     x_trainNeg += result[0]
    #     y_trainNeg += result[1]
    
    # print(len(x_trainNeg), len(y_trainNeg))

    x_trainNeg, y_trainNeg = extractFeaturesNeg(imagesNeg)

    print(len(x_trainNeg), len(y_trainNeg))


    elapsed_time = time.time() - start_time
    print("%.5f" % elapsed_time, 'HOG Neg')
    total_time += elapsed_time
    
    negNo = len(x_trainNeg)

    print("negNo")
    print(negNo)

    start_time = time.time()
    
    x_trainPos, y_trainPos = extractFeaturesPos(imagesPos)

    elapsed_time = time.time() - start_time
    print(elapsed_time, 'HOG Pos')
    total_time += elapsed_time

    # TODO: test if turning into array impacts perf
    x_train = np.array(x_trainNeg + x_trainPos)
    y_train = np.array(y_trainNeg + y_trainPos)

    clf = svm.SVC(C=100, gamma='auto')

    clf.fit(x_train, y_train)

    epochN = 2

    # for each epoch of hard negative mining
    for epoch in range(epochN):
        # train with current set
        # readjust weights
        # weights = {}
        # weights[0] = 1 / (y_train == 0).sum()
        # weights[1] = 1 / (y_train == 1).sum()
        clf.set_params(class_weight='balanced')
        # if last epoch, train with probability enabled
        # to use Non Max Suppresion afterwards
        if epoch == epochN-1:
            clf.set_params(probability=True)

        clf.fit(x_train, y_train)

        def getNegHard(clf, imagesNeg):
            negHard = []
            # slide through images to find hard negatives (false positives)
            for pathImg in imagesNeg:
                img = cv2.imread(pathImg, cv2.IMREAD_GRAYSCALE)
                for win, _ in extractWindows(img):
                    winFeats = windowHog(win)
                    y = clf.predict(win)

                    if y == classPos:
                        # false positive
                        negHard.append(win)

            return negHard, []

        # test classifier with sliding window
        # on the test set to find hard negatives
        # TODO: test which is faster
        # imagesNeg = load.database(
        #     "/home/html/inf/menotti/ci1028-191/INRIAPerson/Train/neg/")
        #
        imagesNeg = load.databaseFilenames(
            "/home/html/inf/menotti/ci1028-191/INRIAPerson/Train/neg/")
        #
        def scrambled(orig):
            dest = orig[:]
            random.shuffle(dest)
            return dest
        imagesNeg = scrambled(list(imagesNeg))
        x_trainNegHard, y_trainNegHard = getNegHard(clf, imagesNeg)
        # add hard negatives on the training set
        x_train += x_trainNegHard
        y_train += y_trainNegHard


    # cv_results = model_selection.cross_validate(clf, x_train, y_train, cv=3)
    # print(cv_results)

    # y_pred = model_selection.cross_val_predict(clf, x_train, y_train, cv=4)
    # conf_mat = metrics.confusion_matrix(y_train, y_pred)
    # print(conf_mat)
