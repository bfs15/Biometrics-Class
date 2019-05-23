
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
from skimage.morphology import skeletonize
import matplotlib.cm
import time

verbose = True

def image_gradient(image):
	image_gradient_x = np.zeros(image.shape)
	image_gradient_y = np.zeros(image.shape)
	image_angle = np.zeros(image.shape)
	image_direction = np.zeros(image.shape)

	return image_angle, image_direction


def sobel_filter(img, axis):
	img = img.astype(np.float)

	if axis == 0:
			# x axis
		kernel = np.array([[1, 0, -1]], dtype=np.float)
	elif axis == 1:
			# y axis
		kernel = np.array([[1], [0], [-1]], dtype=np.float)

	convolved = signal.convolve2d(
			img, kernel, mode='same', boundary='wrap', fillvalue=0)

	return convolved


def gradient(img, blk_sz):
	dx = sobel_filter(img, 0)
	dy = sobel_filter(img, 1)

	dy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
	dx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

	blocks_orientation = np.arctan2(dy, dx)
	blocks_magnitude = 0
	#np.sqrt(dy**2 + dx**2)

	return blocks_orientation, blocks_magnitude


if __name__ == "__main__":
	total_time = 0
	start_time = time.time()

	images = load.database("/home/html/inf/menotti/ci1028-191/INRIAPerson/test_64x128_H96/pos/")
	# images = list(images)

	elapsed_time = time.time() - start_time
	print(elapsed_time, 'load database')
	total_time += elapsed_time

	for image, image_arg in images:
		image = np.array(image)
		print(image.shape)

		cv2.imshow("image"+str(image_arg), image)

		image_angle, image_magnitude = image_gradient(image)
		
		cv2.waitKey(0)
		cv2.destroyAllWindows()

