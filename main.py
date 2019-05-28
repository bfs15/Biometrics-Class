
import load
# import enhance
# import fingerprint
# import compare
# import stats

from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import scipy
import cv2
import sys
import matplotlib.cm
import time

from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm

verbose = True


def sobel_filter(img, axis):
	img = img.astype(np.float)

	if axis == 0:
			# x axis
		kernel = np.array([[1, 0, -1]], dtype=np.float)
	elif axis == 1:
			# y axis
		kernel = np.array([[1], [0], [-1]], dtype=np.float)

	convolved = scipy.signal.convolve2d(
				img, kernel, mode='same', boundary='symm')

	return convolved


def gradient(img):
	"""
	return mag, angle
	"""
	#
	# gx = sobel_filter(img, 0)
	# gy = sobel_filter(img, 1)
	#
	gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
	gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
	#

	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

	return mag, angle


def pyramidCreate(image, levels=3):
	"""
	image : image matrix
	levels : quantity of levels in the pyramid
	returns [imageOriginal, imageLevel1, imageLevel2, ...]
	"""
	pyr = []
	pyr.append(image)

	for level in range(levels):
		image = cv2.pyrDown(image)
		pyr.append(image)

	return pyr


def GradientHistogram(mag, angle, cellSz=8, binSz=9):
	binsShape = (mag.shape[0]//cellSz, mag.shape[1]//cellSz, binSz)
	bins = np.empty(binsShape)

	angle = (angle - 90) % 180
	for bins_y in range(bins.shape[0]):
		for bins_x in range(bins.shape[1]):
			cellMag = mag[(bins_y+0)*cellSz: (bins_y+1)*cellSz,
									 (bins_x+0)*cellSz: (bins_x+1)*cellSz]
			cellAngle = angle[(bins_y+0)*cellSz: (bins_y+1)*cellSz,
									 (bins_x+0)*cellSz: (bins_x+1)*cellSz]

			hist, bin_edges = np.histogram(cellAngle, bins=binSz, range=(0, 180), normed=None,
											 weights=cellMag, density=None)
			
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


def extractWindows(image, stride=8, winSz=(128,64)):
	image = cv2.copyMakeBorder(image, winSz[0]//2, winSz[0]//2, winSz[1]//2, winSz[1]//2, cv2.BORDER_REPLICATE)
	for win_y in range(0, image.shape[0]-winSz[0], stride):
		for win_x in range(0, image.shape[1]-winSz[1], stride):
			yield image[win_y : win_y + winSz[0]  ,  win_x : win_x + winSz[1]]


if __name__ == "__main__":
	total_time = 0

	images = load.database("/home/html/inf/menotti/ci1028-191/INRIAPerson/test_64x128_H96/pos/")
	imagesNeg = load.database("/home/html/inf/menotti/ci1028-191/INRIAPerson/Train/neg/")
	x_train = []
	y_train = []

	for image, image_arg in imagesNeg:
		image = np.array(image)

		for imageLevel in pyramidCreate(image):
			for window in extractWindows(imageLevel):
				features = windowHog(window)
				x_train.append(features)
				y_train.append(1)
				if(np.isnan(features).any()):
					print("window.shape")
					print(window.shape)
					print(features)
				sys.stdout.flush()
	
	

	for image, image_arg in images:
		image = np.array(image)

		start_time = time.time()

		features = windowHog(image)

		elapsed_time = time.time() - start_time
		print(elapsed_time, 'HOG')
		total_time += elapsed_time

		x_train.append(features)
		y_train.append(0)

		cv2.imshow("image"+str(image_arg), image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# clf = sklearn.svm.SVC()
	# # clf = sklearn.svm.OneClassSVM()
	# clf.fit(x_train, y_train)

