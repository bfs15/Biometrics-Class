#! python2

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
from skimage import data, exposure
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection

verbose = True


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


def extractWindows(image, stride=128, winSz=(128, 64)):
    image = cv2.copyMakeBorder(image, winSz[0]//4, winSz[0]//4, winSz[1]//4, winSz[1]//4, cv2.BORDER_REPLICATE)
    for win_y in range(0, image.shape[0]-winSz[0], stride):
        for win_x in range(0, image.shape[1]-winSz[1], stride):
            yield image[win_y : win_y + winSz[0]  ,  win_x : win_x + winSz[1]]


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
            

def extractFeaturesBackground(imagesFalse):
    x_train = []
    y_train = []
    for image_arg, image in enumerate(imagesFalse):
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
                y_train.append(-1)  # negative class
                if(np.isnan(features).any()):
                    print("- warning: nan features")
                    print("window.shape")
                    print(window.shape)
                    print(features)
                    sys.stdout.flush()

    return x_train, y_train


def extractFeaturesTrue(imagesTrue):
    x_train = []
    y_train = []
    for image_arg, image in enumerate(imagesTrue):
        image = np.array(image)

        features = windowHog(image)

        x_train.append(features)
        y_train.append(1)

        # cv2.imshow("image"+str(image_arg), image)
        # cv2.waitKey(64)
        # cv2.destroyAllWindows()
        
    return x_train, y_train

if __name__ == "__main__":
    total_time = 0

    imagesTrue = load.database(
        "/home/html/inf/menotti/ci1028-191/INRIAPerson/70X134H96/Test/pos")
    imagesFalse = load.database("/home/html/inf/menotti/ci1028-191/INRIAPerson/Train/neg/")

    start_time = time.time()

    x_trainFalse, y_trainFalse = extractFeaturesBackground(imagesFalse)

    elapsed_time = time.time() - start_time
    print("%.5f" % elapsed_time, 'HOG False')
    total_time += elapsed_time
    
    falseNo = len(x_trainFalse)

    print("falseNo")
    print(falseNo)

    start_time = time.time()
    
    x_trainTrue, y_trainTrue = extractFeaturesTrue(imagesTrue)

    elapsed_time = time.time() - start_time
    print(elapsed_time, 'HOG True')
    total_time += elapsed_time

    x_train = x_trainFalse + x_trainTrue
    y_train = y_trainFalse + y_trainTrue

    clf = svm.SVC(C=100, gamma='auto')

    # clf = sklearn.svm.OneClassSVM()
    # clf.fit(x_train, y_train)

    # cv_results = model_selection.cross_validate(clf, x_train, y_train, cv=3)
    # print(cv_results)

    y_pred = model_selection.cross_val_predict(clf, x_train, y_train, cv=4)
    conf_mat = metrics.confusion_matrix(y_train, y_pred)
    print(conf_mat)
