
from __future__ import print_function

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
import os
import matplotlib.cm
import time
import itertools
import random
import pickle
import joblib

import multiprocessing
from numba import jit
import psutil

import skimage.feature
import sklearn
from skimage import data, exposure
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
import sklearn.ensemble

classPos = 1
classNeg = 0
NMSThresh = 0.5
processorN = 4
boxPredictionThreshold = 0.5
curveStep = 0.01
cpuCount = multiprocessing.cpu_count() # // 2
# pyramid
lvlsUp = 2
lvlsDown = 1
pyrScale = 0.20
imgDownscale = 0.5


def limitProcessPriority():
	"is called at every process start"
	p = psutil.Process(os.getpid())
	# set to lowest priority, this is windows only, on Unix use ps.nice(19)
	p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	n = len(l)//n + 1
	for i in range(0, len(l), n):
		yield list(l[i:i + n])


def chunkSplit(l, n):
	"""Yield n successive evenly-sized chunks from l."""
	step = len(l)//n + 1
	for i in range(0, len(l), step):
		yield list(l[i:i + step])


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


def pyramidCreate(image, levelsUp=lvlsUp, levelsDown=lvlsDown, scale=pyrScale):
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
	# rotate angles 90 degrees and put them on the first and second quadrant
	angle = (angle - 90) % 180
	for bins_y in range(bins.shape[0]):
		for bins_x in range(bins.shape[1]):
			# for each cell (y,x), slice the mag and angle matrixes
			cellMag = mag[(bins_y+0)*cellSz: (bins_y+1)*cellSz,
						  (bins_x+0)*cellSz: (bins_x+1)*cellSz].flatten()
			cellAngle = angle[(bins_y+0)*cellSz: (bins_y+1)*cellSz,
							  (bins_x+0)*cellSz: (bins_x+1)*cellSz].flatten()
			## tri-linear histogram
			# initialize histogram
			hist = np.zeros((binSz))
			for angleEl, magEl in zip(cellAngle, cellMag):
				# for each pixel in the cell (its angle and mag)
				# calculate continuous position in the histogram
				pos = angleEl/(180/binSz) - 0.5
				# calculate discrete position in the histogram
				index = int(math.floor(pos)) % binSz
				# get how close it is from the bin[index]
				weight = abs(pos - int(pos))
				# weight the magnitude by how close to the bin position it is
				hist[index] = (1-weight)*magEl
				hist[(index+1) % binSz] = (weight)*magEl
			## histogram
			# hist, bin_edges = np.histogram(cellAngle, bins=binSz, range=(0, 180), normed=None, weights=cellMag, density=None)
			##
			# save the cell histogram in this matrix
			bins[bins_y, bins_x] = hist
	# return the cell histogram matrix
	return bins


@jit(nopython=True)
def normalizeHistogram(bins, blkSz=2, stride=1, eps=1e-6):
	"""
	bins : shape (imgHeight//cellSz, imgHeight//cellSz, histogramSize)
	blkSz : block size in cells, a block will be a square of (blkSz x blkSz) cells
	stride : stride in cells, ex: blkSz=2 stride=1 has an overlap of 50%
	"""
	binsNormShape = ((bins.shape[0] - blkSz+1)//stride,
					 (bins.shape[1] - blkSz+1)//stride, bins.shape[-1]*(blkSz**2))
	binsNorm = np.empty(binsNormShape)

	for bins_y in range(binsNorm.shape[0]):
		for bins_x in range(binsNorm.shape[1]):
			block = bins[(bins_y*stride): (bins_y*stride) + blkSz,
						 (bins_x*stride): (bins_x*stride) + blkSz]

			blockFlat = block.flatten()
			L2Norm = np.linalg.norm(blockFlat) + eps
			binsNorm[bins_y, bins_x] = blockFlat / L2Norm
			# if(np.isnan(binsNorm[bins_y, bins_x]).any()):
			#     binsNorm[bins_y, bins_x] = np.zeros(binsNormShape)

	return binsNorm


def windowHog(image, blkSz=2, stride=1, cellSz=8, binSz=9):
	"""
	image : image window
	return feature vector of the window
	"""
	mag, angle = gradient(image)
	bins = GradientHistogram(mag, angle, cellSz, binSz)
	binsNorm = normalizeHistogram(bins, blkSz, stride)
	## To display gradient to user ##
	# fd, hog_image = skimage.feature.hog(image, orientations=8, pixels_per_cell=(cellSz, cellSz), cells_per_block=(blkSz, blkSz), block_norm='L2', visualise=True)
	# # show resulting image
	# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
	# cv2.imshow("hog_image", hog_image_rescaled)
	# cv2.waitKey()
	## cv2 style gradient ##
	# winSize = (64, 128)
	# blockSize = (blkSz*cellSz, blkSz*cellSz)
	# blockStride = (8, 8)
	# cellSize = (cellSz, cellSz)
	# hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, binSz)
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
			yield (image[winBegY:winEndY,  winBegX: winEndX],  np.array([winBegY, winBegX, winEndY, winEndX]))


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
		image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		# skip some images for faster testing
		if not image_arg % 12 == 0:
			continue

		image = np.array(image)
		# for imageLevel in pyramidCreate(image, 4, 3, 0.1):
		for imageLevel in pyramidCreate(image):
			for window in extractWindowsRandom(imageLevel, 1):
				# show extracted window
				# cv2.imshow("win", window)
				# cv2.waitKey(64)
				features = windowHog(window)
				x_train.append(features)
	# class label array
	x_train = np.array(x_train)
	y_train = np.full(x_train.shape[0], classNeg)
	return x_train, y_train


def extractFeaturesPos(windowsPos):
	x_train = []
	y_train = []
	for image_arg, window in enumerate(windowsPos):
		features = windowHog(window)
		x_train.append(features)

		# cv2.imshow("image"+str(image_arg), image)
		# cv2.waitKey(64)
		# cv2.destroyAllWindows()
	# class label array
	x_train = np.array(x_train)
	y_train = np.full(x_train.shape[0], classPos)
	return x_train, y_train


@jit(nopython=True)
def nonMaxSuppresionIdxs(boxes, probas, overlapThresh):
	"""
	boxes: 2darray of shape (n, 4) / n is the number of boxes
	probas: 1darray of shape (n), each element is the probability of the corresponding box
	overlapThresh: if boxes exceed this threshold of overlap with the maximum box, it is suppressed
	"""
	# list of max boxes indexes
	maxBoxIdxs = []
	maxBoxProbas = []
	# coordinates of the bounding boxes
	begY = boxes[:, 0]
	begX = boxes[:, 1]
	endY = boxes[:, 2]
	endX = boxes[:, 3]
	# area of the bounding boxes
	area = (endY - begY) * (endX - begX)
	# sort boxes by the probabilities
	argProbas = np.argsort(probas)
	# loop removing boxes from the idx list, until no boxes remain
	while len(argProbas) > 0:
		# get the max proba idx in the idxs list (last in a sorted array)
		idxMax = argProbas[-1]
		# save max box idx in the list (to return them later)
		maxBoxIdxs.append(idxMax)
		maxBoxProbas.append(probas[idxMax])
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
		overlH = np.maximum(0, overlEndY - overlBegY)
		overlW = np.maximum(0, overlEndX - overlBegX)
		# overlap ratio
		overlArea = overlH * overlW
		overlRatio = (overlArea) / area[argProbas[:-1]]
		# box idxs which overlap with the max box is over the threshold
		overlBoxIdxs = np.where(overlRatio > overlapThresh)[0].astype(np.int32)
		# delete boxes that overlap and the max proba box
		lastList = np.array([(argProbas.shape[0])-1], np.int32)
		idxsDelete = np.concatenate((overlBoxIdxs, lastList))
		argProbas = np.delete(argProbas, idxsDelete)

	# return boxes from max box idxs
	return maxBoxIdxs, maxBoxProbas


def nonMaxSuppresion(boxes, probas, overlapThresh):
	if len(boxes) == 0:
		# no boxes
		return np.empty([0,4]), np.array([])
	# if the bounding boxes are integers
	# convert them to floats, float division is faster
	if boxes.dtype.kind == "i":
		boxes = boxes.astype(np.float32)

	maxBoxIdxs, maxBoxProbas = nonMaxSuppresionIdxs(boxes, probas, overlapThresh)
	# return boxes from max box idxs
	return boxes[maxBoxIdxs], np.array(maxBoxProbas)

def nonMaxSuppresionTest():
	boxes = np.array([[1, 2, 3, 4], [1, 2, 3, 5], [1+100, 2 + 100,
					3 + 100, 4 + 100], [1+100, 2 + 100, 3 + 100, 5 + 100]])
	probas = np.array([.50, 1.0,  0.9,0.4])
	boxes, probas = nonMaxSuppresion(boxes, probas, 0.5)
	print("\nreturn:")
	for box, prob in zip (boxes, probas):
		print("{0:0.3}".format(prob), box)
	"""
	1.0 [1. 2. 3. 5.]
	0.9 [101. 102. 103. 104.]
	"""

def extractFeats(img):
	# array of probability of predictions for windows
	peopleProb = []
	# array of the position of windows
	peopleWin = []
	winSz = (128, 64)
	imgFeats = []
	print("predicting image of size: ", img.shape)
	for argLvl, imgLvl in enumerate(pyramidCreate(img, lvlsUp, lvlsDown, pyrScale)):
		if argLvl > lvlsUp:
			# negative levels, down in the pyramid
			argLvl = argLvl-(lvlsUp+lvlsDown+1)

		start_time = time.time()
		print("pyramid lvl: ", argLvl, flush=True)

		# if scale decreases, window increases in relative size
		# e.g. the window in the 0.5 scaled image has double the size (1/0.5)
		winScale = 1/(1+pyrScale*argLvl)
		for win, winBoxLvl in extractWindows(imgLvl):
			winFeats = windowHog(win)
			imgFeats.append(winFeats)
			winBox = winBoxLvl * winScale
			peopleWin.append(winBox)

		elapsed_time = time.time() - start_time
		print("%.5f" % elapsed_time, 'windowsNo=', len(imgFeats))
	# Save number of windows for FFPW calculation
	windowNo = len(peopleWin)
	# convert to np array for slicing
	peopleWin = np.array(peopleWin)
	return imgFeats, peopleWin, windowNo


def predictFeatures(clf, imgFeats, peopleWin):
	# predict probabilities for all extracted windows
	start_time = time.time()
	""" # Parallel
	pred = np.array(parallelListFunc(clf.predict_proba, imgFeats))
	"""  # Sequential
	# """
	print("Started predicting probabilities, featNo=", len(imgFeats),  flush=True)
	print("Progress: ", end=" ",  flush=True)
	chunks = chunkSplit(imgFeats, 20)
	predictions = []
	for chunkNo, chunk in enumerate(chunks):
		pred = np.array(clf.predict_proba(chunk))
		predictions.append(pred)
		print(" {:0.2f}".format((chunkNo+1)/20), end=" ",  flush=True)
	print()
	pred = np.concatenate(predictions)
	"""
	pred = np.array(clf.predict_proba(imgFeats))
	# """

	elapsed_time = time.time() - start_time
	print("%.5f" % elapsed_time, 'prediction', len(pred))
	# Get only positive prediction probability
	probas = pred[:, classPos]
	# select indexes over threshold
	idxs = np.where(probas > boxPredictionThreshold)
	peopleWin = peopleWin[idxs]
	peopleProb = np.array(probas[idxs])
	# from the box in this pyrLvl, get the real widow
	print("Found windows: ", peopleWin.shape, flush=True)
	return np.array(peopleWin), np.array(peopleProb)


def predictImage(clf, img):
	"""
	Given classifier and image, returns a array of bounding boxes (boxesNo,4)
	boxesPred = predictImage(clf, img)
	"""
	# Extract features from all windows in image
	imgFeats, peopleWin, windowNo = extractFeats(img)
	# From features, use classifier to predict probabilities
	peopleWin, peopleProb = predictFeatures(clf, imgFeats, peopleWin)
	# non maximum suppresion
	peopleBoxes, peopleProb = nonMaxSuppresion(peopleWin, peopleProb, NMSThresh)
	print("Found windows Max: ", peopleBoxes.shape, flush=True)

	return peopleBoxes, peopleProb, windowNo


# TODO: @jit(nopython=True)
def compareBoxes(boxesTrue, boxesPred, overlapThresh):
	"""
	boxesTrue: true boxes
	boxesPred: predicted boxes
	# boxes: 2darray of shape (n, 4) / n is the number of boxes
	overlapThresh: if boxes exceed this threshold of overlap, it is considered the same box
	"""
	# initialize stats
	truePos = 0
	falseNeg = 0
	# if the bounding boxes are integers
	# convert them to floats, float division is faster
	if boxesTrue.dtype.kind == "i":
		boxesTrue = boxesTrue.astype(np.float)
	if boxesPred.dtype.kind == "i":
		boxesPred = boxesPred.astype(np.float)
	# list of max boxes indexes
	maxBoxIdxs = []
	# coordinates of the bounding boxes
	begY = 0
	begX = 1
	endY = 2
	endX = 3
	trueBegY = boxesTrue[:, begY]
	trueBegX = boxesTrue[:, begX]
	trueEndY = boxesTrue[:, endY]
	trueEndX = boxesTrue[:, endX]
	# area of the bounding boxes
	area = (trueEndY - trueBegY + 1) * (trueEndX - trueBegX + 1)
	# loop removing boxes from the idx list, until no boxes remain
	for idx in range(len(boxesTrue)):
		# print("idx", idx, "boxesTrue[idx]", boxesTrue[idx], flush=True)
		# print("len(boxesPred)", len(boxesPred), flush=True)
		if len(boxesPred) < 1:
			# no predicted box overlaps with this true one
			falseNeg += 1
			continue
		# calculate overlap:
		# get bot-rightmost beginning (x, y) coordinates of both the boxes
		# and the top-leftmost ending (x, y) coordinates of both the boxes
		# the area of overlap is the area of the box of those coordinates
		# use np.maximum to calculate overlap for every box, is the same as:
		# max(begY[idxMax] - begY[idx]) for idx in argProbas[:-1] (every box except the max one)
		overlBegX = np.maximum(trueBegY[idx], boxesPred[:, begY])
		overlBegY = np.maximum(trueBegX[idx], boxesPred[:, begX])
		overlEndX = np.minimum(trueEndY[idx], boxesPred[:, endY])
		overlEndY = np.minimum(trueEndX[idx], boxesPred[:, endX])
		# width and height of the overlap box
		# the normal calculation (end-beg+1) can be negative in the cases boxes don't overlap
		overlH = np.maximum(0, overlEndY - overlBegY + 1)
		overlW = np.maximum(0, overlEndX - overlBegX + 1)
		# overlap ratio
		overlArea = overlH * overlW
		overlRatio = (overlArea) / area[idx]
		# print("overlH",overlH, flush=True)
		# print("overlW",overlW, flush=True)
		# print("overlRatio", overlRatio, flush=True)
		# true positive if the box with most overlap is over threshold
		idxMaxOverl = np.argmax(overlRatio)
		if overlRatio[idxMaxOverl] > overlapThresh:
			# a predicted box overlaps with this true one
			truePos += 1
			# delete the box registered
			boxesPred = np.delete(boxesPred, [idxMaxOverl], axis=0)
		else:
			# no predicted box overlaps with this true one
			falseNeg += 1
	# false positives = number of boxes not paired with any true one (remaining)
	falsePos = boxesPred.shape[0]
	# return stats
	return truePos, falsePos, falseNeg


def evaluateImage(boxesPred, peopleProb, boxesTrue, thresholds=np.arange(boxPredictionThreshold, 1.0-curveStep, curveStep)):
	"""
	Given classifier, image and true labels
	returns statistics of the predictions with multiple thresholds
	return truePosList, falsePosList, falseNegList, threshList
	"""
	stats = []
	for threshold in thresholds:
		boxesThresh = boxesPred[np.where(peopleProb > threshold)]
		truePos, falsePos, falseNeg = compareBoxes(
			boxesTrue, boxesThresh, 0.5)
		# missRate =  falseNegative / conditionPositive
		missRate = falseNeg / len(boxesTrue)

		stats.append([truePos, falsePos, falseNeg, threshold])
	truePosList, falsePosList, falseNegList, threshList = list(zip(*stats))
	return truePosList, falsePosList, falseNegList, threshList


def tupleListToMultipleLists(tupleList):
	"""
	Given a list in the format: [0:([x..],[y..]) ... N:([x..],[y..])]
	returns concatenated list: x:[0: ..., ... , N: ...], y:[0: ..., ... , N: ...]
	"""
	return list(map(np.concatenate, zip(*tupleList)))


def splitListN(a, n):
	k, m = divmod(len(a), n)
	return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def parallelListFunc(fun, itemList, isTupleList=False, *argv):
	"""
	Processes function as if calling fun(itemList, *argv)
	Divides itemList evenly across and calls fun in parallel
	fun must have a list argument and return a list

	isTupleList: if the returned list is a list of tuples
	"""
	# Split items to process into chunks (evenly)
	itemChunks = list(chunkSplit(itemList, cpuCount))
	print('dividing work to cpus')
	for chunk in itemChunks:
		print(len(chunk))
	# Create Pool with the number of procs detected
	p = multiprocessing.Pool(cpuCount)
	# poolResult = [returnFun0, ... , returnFunProcNo]
	poolResult = p.map(fun, itemChunks)
	# each list element of poolResult is a thread result
	# poolResult = [(x:[1],y:[2]),(x:[3],y:[4])]
	# print("poolResult.shape", np.array(list(poolResult)).shape)
	# print("poolResult.shape", np.array(list(poolResult[0][0])).shape)
	# Join pool into one list
	if isTupleList:
		poolResult =  tupleListToMultipleLists(poolResult)
	else:
		poolResult = list(map(np.concatenate, poolResult))
	# for res in list(poolResult):
	# 	print(res.shape)
	# print(type(poolResult))
	# print(type(list(poolResult)))
	# for el in list(poolResult):
	# 	print("???")
	# 	print(type(el),el.shape)
	return poolResult


def funUnpack(args):
	"""
	Use it to call a function with one argument, for example
	multiprocessing.Pool().map(funUnpack, arguments)
	"""
	fun = args[0]
	return fun(*args[1:])


def parallelListFuncArgv(fun, itemList, isTupleList=False, *argv):
	"""
	Processes function as if calling fun(itemList, *argv), you will need to make an unpacker
	Divides itemList evenly across and calls fun in parallel
	fun must have a list argument and return a list

	isTupleList: if the returned list is a list of tuples
	"""
	def splitListN(a, n):
		k, m = divmod(len(a), n)
		return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
	# Split items to process into chunks (evenly)
	itemChunks = list(splitListN(itemList, cpuCount))
	# Create Pool with the number of procs detected
	p = multiprocessing.Pool(cpuCount)
	# function unpacker, since p.map accepts only 1 argument
	# list of arguments, len(arguments) == cpu_count()
	arguments = []
	# for each cpu, make its arguments
	for idx in range(cpuCount):
		# put the function and a chunk into the arguments
		arguments.append([fun, itemChunks[idx]])
		# pass on the rest of the arguments
		for arg in argv:
			arguments[idx].append(arg)
	# Pool functionUnpacker with arguments
	poolResult = p.map(funUnpack, arguments)
	# poolResult = [returnFun0, ... , returnFunProcNo]
	# each return is a list (or a tuple of lists)
	# Join pool into one list
	if isTupleList:
		return tupleListToMultipleLists(poolResult)
	else:
		return map(np.concatenate, poolResult)


def loadFeats(pathFeats, classNo):
	print("Reading class {} from disk...".format(classNo))
	sys.stdout.flush()
	x_train = np.load(pathFeats, allow_pickle=True)
	y_train = np.full(x_train.shape[0], classNo)
	return x_train, y_train


def getNegHard(imagesNeg, clf, featsLimitNo):
	"""
	Given classifier, returns hard negatives from an image set
	clf: classifier
	imagesNeg: images with only negative conditions
	return negHardX(featsLimitNo, featuresDimension), negHardY(featsLimitNo,)
	"""
	negHardX = None
	# slide through images to find hard negatives (false positives)
	while True:
		for imgIdx, pathImg in enumerate(imagesNeg):
			img = cv2.imread(pathImg, cv2.IMREAD_GRAYSCALE)
			img = np.array(img)
			# list of dimensions (windows, features)
			winFeats = []
			# for imgLvl in pyramidCreate(img, 4, 3, 0.1):
			for imgLvl in pyramidCreate(img):
				# skip some images for faster testing
				# if not imgIdx % 100 == 0:
				#     continue
				#
				for win in extractWindowsRandom(imgLvl, 4):
					winFeat = windowHog(win)
					winFeats.append(winFeat)
			# predict windows extracted from image
			y = clf.predict(winFeats)
			# select windows which are false positievs
			winFeats = np.array(winFeats)
			falsePos = winFeats[np.where(y == classPos)]
			try:
				negHardX = np.concatenate([negHardX, falsePos])
			except ValueError as err:
				# zero-dimensional arrays cannot be concatenated
				negHardX = falsePos
			# if we have enough features, break out of loop
			if negHardX.shape[0] > featsLimitNo:
				break
		# if we have enough features, break out of loop
		if negHardX.shape[0] > featsLimitNo:
			break
	negHardY = np.full(negHardX.shape[0], classNeg)
	return negHardX, negHardY


if __name__ == "__main__":
	import argparse
	## Instantiate the parser
	parser = argparse.ArgumentParser(
		description='script description\n'
		'python ' + __file__ + ' arg1 example usage',
		formatter_class=argparse.RawTextHelpFormatter)
	# arguments
	parser.add_argument('-v', '--verbose', action='store_true',
						help='verbose')
	parser.add_argument('-nc', '--noCache', action='store_true',
						help='ignores cached data on disk')
	## Parse arguments
	args = parser.parse_args()
	# hopefully your pc doesn't freeze
	limitProcessPriority()

	timestr = time.strftime("%Y%m%d-%H%M")
	## load files
	# filepath variables
	# prefix     = "/home/html/inf/menotti/ci1028-191/INRIAPerson/"
	prefix     = "./DB/INRIAPerson/"
	dbTrain    = prefix + "/Train"
	dbTrainPos = prefix + "/70X134H96/Test/pos"
	dbTrainNeg = prefix + "/Train/neg"
	dbTest     = prefix + "/Test"
	dbTestPos  = prefix + "/Test/pos"
	dbTestNeg  = prefix + "/Test/neg"
	dirDB      = "DB"
	extNp = ".npy"
	x_trainNegPathname = dirDB + '/' + 'featsNegFile'
	x_trainNegPath = x_trainNegPathname + extNp
	x_trainPosPathname = dirDB + '/' + 'featsPosFile'
	x_trainPosPath = x_trainPosPathname + extNp
	curveImgPathname = "curve"
	# list of filepaths
	imgPathsPosTrain, imgPathsNegTrain, boxesPosTrain, windowsPosTrain = load.INRIAPerson(dbTrain)
	imgPathsPosTest, imgPathsNegTest, boxesPosTest, windowsPosTest = load.INRIAPerson(dbTest)
	# exit()
	## load files
	# """"" # Train model
	# ### Read positive samples ###
	# start_time = time.time()

	# if os.path.isfile(x_trainPosPath) and not args.noCache:
	# 	# read features from disk
	# 	x_trainPos, y_trainPos = loadFeats(x_trainPosPath, classPos)
	# else:
	# 	# extract windows from the positive test folder
	# 	# """ # Parallel
	# 	x_trainPos, y_trainPos = parallelListFunc(
	# 		 extractFeaturesPos, list(windowsPosTrain), isTupleList=True)
	# 	""" # Sequential
	# 	x_trainPos, y_trainPos = extractFeaturesPos(windowsPosTrain)
	# 	# """
	# 	# save features to disk for faster testing
	# 	np.save(x_trainPosPath, x_trainPos)

	# elapsed_time = time.time() - start_time
	# print("%.5f" % elapsed_time, 'Feats Pos')
	# ### Read negative samples ###
	# start_time = time.time()
	# # if cache file exists and no argument against
	# if os.path.isfile(x_trainNegPath) and not args.noCache:
	# 	# read features from disk
	# 	x_trainNeg, y_trainNeg = loadFeats(x_trainNegPath, classNeg)
	# else:
	# 	# extract features from images
	# 	# """ # Parallel
	# 	x_trainNeg, y_trainNeg = parallelListFunc(
	# 		extractFeaturesNeg, list(imgPathsNegTrain), isTupleList=True)
	# 	""" # Sequential
	# 	# x_trainNeg, y_trainNeg = extractFeaturesNeg(list(imgPathsNeg))
	# 	# """
	# 	# save features to disk for faster testing
	# 	np.save(x_trainNegPath, x_trainNeg)

	# elapsed_time = time.time() - start_time
	# print("%.5f" % elapsed_time, 'Feats Neg')
	# # concatenate positive and negative data into the train set
	# print("x_trainNeg.shape", x_trainNeg.shape)
	# print("x_trainPos.shape", x_trainPos.shape)
	# x_train = np.concatenate([x_trainNeg, x_trainPos])
	# y_train = np.concatenate([y_trainNeg, y_trainPos])
	# ### Hard negative mining
	# # clf = svm.SVC(C=1, gamma='auto', class_weight='balanced')
	# n_estimators = cpuCount
	# clf = sklearn.ensemble.BaggingClassifier(svm.SVC(kernel='linear', C=100, gamma='auto', class_weight='balanced', probability=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)
	# # first fit to all the postive data and random negative windows
	# # clf.fit(x_train, y_train)
	# # number of epochs of Hard Negative Mining
	# epochN = 2
	# # epochN = 5
	# # for each epoch of hard negative mining
	# for epoch in range(epochN):
	# 	### Fit classifier to current set
	# 	start_time = time.time()

	# 	print("epoch: ", epoch)
	# 	sys.stdout.flush()
	# 	## train with current set
	# 	# if last epoch, train with probability enabled
	# 	# to use Non Max Suppresion afterwards
	# 	if epoch == epochN-1:
	# 		print("last epoch, probability enabled")
	# 		# clf.set_params(probability=True)
	# 	# train
	# 	clf.fit(x_train, y_train)
	# 	# Save model to disk
	# 	modelBasename = 'classifier'
	# 	modelExt = '.joblib'
	# 	modelPath = modelBasename + '+' + time.strftime("%Y%m%d-%H%M") + modelExt
	# 	print("saving model to ", modelPath)
	# 	joblib.dump(clf, modelPath)

	# 	elapsed_time = time.time() - start_time
	# 	print("%.5f" % elapsed_time, 'epoch', epoch, 'finished')
	# 	sys.stdout.flush()
	# 	### Get hard examples
	# 	start_time = time.time()

	# 	if epoch == epochN-1:
	# 		# if last epoch, no need to hardmine
	# 		break
	# 	# use classifier on the test set to find hard negatives
	# 	# scramble background input
	# 	def scrambled(orig):
	# 		dest = orig[:]
	# 		random.shuffle(dest)
	# 		return dest
	# 	imgPathsNegShuffled = scrambled(list(imgPathsNegTrain))
	# 	# get hard negatives
	# 	print("trying to get hard negatives:", x_train.shape[0])
	# 	""" # Parallel
	# 	x_trainNegHard, y_trainNegHard = parallelListFuncArgv(getNegHard,
	# 		imgPathsNegShuffled, True, clf, x_train.shape[0] / cpuCount)
	# 	""" # Sequential
	# 	x_trainNegHard, y_trainNegHard = getNegHard(imgPathsNegShuffled, clf, x_train.shape[0])
	# 	# """
	# 	# add hard negatives on the training set
	# 	print("hard negatives no:", x_trainNegHard.shape)
	# 	x_train = np.concatenate([x_train, x_trainNegHard])
	# 	y_train = np.concatenate([y_train, y_trainNegHard])

	# 	x_trainNegPathEpoch = x_trainNegPathname + '+' + time.strftime("%Y%m%d-%H%M") + extNp
	# 	print("saving neg to ", x_trainNegPathEpoch)
	# 	np.save(x_trainNegPathEpoch, x_trainNeg)

	# 	elapsed_time = time.time() - start_time
	# 	print("%.5f" % elapsed_time, 'epoch', epoch, 'hard examples')
	# 	sys.stdout.flush()
	# # end hard negative mining
	# """"" # Train model
	# """"" # Load a model
	clf = joblib.load('classifier+20190609-1357.joblib')
	# """"" # Load a model

	thresholds = np.arange(boxPredictionThreshold, 1.0-curveStep, curveStep)
	truePosTotal = np.zeros(thresholds.shape)
	falsePosTotal = np.zeros(thresholds.shape)
	falseNegTotal = np.zeros(thresholds.shape)
	missRateTotal = np.zeros(thresholds.shape)
	imgTotalNo = len(boxesPosTest)
	imgNo = 0
	imgPathsPosTest = list(imgPathsPosTest)
	## Print for user
	# for imgPath in imgPathsPosTest:
	# 	print(imgPath)
	# load.printBoxStats(boxesPosTest)
	# plt.show()
	try:
		for imgPath, boxesTrue in zip(imgPathsPosTest, boxesPosTest):
			print(imgPath, flush=True)
			print(boxesTrue, flush=True)
			imgTrue = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(imgTrue, None, fx=imgDownscale, fy=imgDownscale)
			# Get classifier preditction boxes and their probabilities
			boxesPred, boxesProbas, windowNo = predictImage(clf, img)
			# fix axes
			boxesThreshCopy = boxesPred.copy()
			boxesThreshCopy[:, 1] = boxesPred[:, 0]
			boxesThreshCopy[:, 0] = boxesPred[:, 1]
			boxesThreshCopy[:, 3] = boxesPred[:, 2]
			boxesThreshCopy[:, 2] = boxesPred[:, 3]
			boxesPred = boxesThreshCopy.astype(np.int32)
			# fix scale
			boxesPred = (boxesPred * 1.0/imgDownscale).astype(np.int32)
			# Compare true boxes with predictions with with a varying threshold
			result = evaluateImage(boxesPred, boxesProbas, boxesTrue, thresholds)
			# Make result stats np arrays
			truePosList, falsePosList, falseNegList, threshList = map(np.array, result)
			# Calculate missRateList from raw stats
			missRateList = falseNegList / (truePosList + falseNegList + 1.0e-07)
			# """ # Display for user
			print(boxesPred[np.where(boxesProbas > 0.7)])
			for box in boxesPred[np.where(boxesProbas > 0.7)]:
				cv2.rectangle(imgTrue, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 3)
			cv2.imwrite("boxes.png", imgTrue)
			# cv2.waitKey(10)
			# """ # Display for user
			# Accumulate stats
			truePosTotal += truePosList
			falsePosTotal += falsePosList
			falseNegTotal += falseNegList
			missRateTotal += missRateList
			# count images
			imgNo += 1
			# print progress and partial stats
			print(">>> {:0.4f} of images processed ".format(
				imgNo/imgTotalNo), time.strftime("%Y%m%d-%H%M"))
			print("falsePosList =", falsePosList)
			print("missRateList =", missRateList)
			print("falsePosTotal=", falsePosTotal / imgNo)
			print("missRateTotal=", missRateTotal / imgNo)
	except KeyboardInterrupt as err:
		print("\nStopped after imgNo=", imgNo)
		pass
	# Get mean values per image
	truePosPerImg =  truePosTotal / imgNo
	falsePosPerImg =  falsePosTotal / imgNo
	falseNegPerImg =  falseNegTotal / imgNo
	missRate = missRateTotal / imgNo


	def pltCurve(valuesX, valuesY, labelX, labelY, scale='log', limits=[0.01, 1.00, 0.01, 1.00]):
		"""
		plots a curve, where 
		ex: Given false positive and false negative rates, produce a DET Curve.
		import matplotlib.pyplot as plt
		pltCurve(fpsList, fnsList, "false positive rate", "false negative rate")
		plt.show()
		plt.savefig(pathCurvePic)
		"""
		import matplotlib.pyplot as plt
		import matplotlib.ticker
		plt.style.use('dark_background')
		axis_min = min(valuesX[0], valuesY[-1])
		fig, ax = plt.subplots()
		plt.xlabel(labelX)
		plt.ylabel(labelY)
		plt.plot(valuesX, valuesY, '-|')
		plt.grid(linestyle='--', alpha=0.5)
		plt.xscale(scale)
		plt.yscale(scale)
		ax.get_xaxis().set_major_formatter(
					matplotlib.ticker.FuncFormatter(lambda y, _: '{:0.2f}'.format(y)))
		ax.get_yaxis().set_major_formatter(
					matplotlib.ticker.FuncFormatter(lambda y, _: '{:0.2f}'.format(y)))
		# ticks X
		ticks_to_useX = np.logspace(
			np.log10(limits[0]), np.log10(limits[1]), num=10, base=10)
		ticks_to_useX = np.round(ticks_to_useX, 2)
		ax.set_xticks(ticks_to_useX)
		# ticks Y
		ticks_to_useY = np.logspace(
			np.log10(limits[2]), np.log10(limits[3]), num=10, base=10)
		ticks_to_useY = np.round(ticks_to_useY, 2)
		# ax.tick_params(direction='out', length=6, width=2, colors='r', grid_color='r', grid_alpha=0.5)
		ax.set_yticks(ticks_to_useY)
		#
		plt.axis(limits)
		# plt.show()
		return


	# Plot the curve FPPI vs missRate
	pltCurve(falsePosPerImg, missRate, "falsePosPerImg", "missRate",
			 scale='log', limits=[0.01, np.amax(falsePosPerImg), 0.01, 1.00])
	# Save curve img with timestamp
	plt.savefig(curveImgPathname + '+' + timestr + ".png")

	# cv_results = model_selection.cross_validate(clf, x_train, y_train, cv=3)
	# print(cv_results)

	# y_pred = model_selection.cross_val_predict(clf, x_train, y_train, cv=4)
	# conf_mat = metrics.confusion_matrix(y_train, y_pred)
	# print(conf_mat)
