
import glob
import os
import sys
import numpy as np

import cv2

def database(path_in, filetypeExt_in="png"):
	for imgPath in sorted(glob.iglob(os.path.join(path_in, "*." + filetypeExt_in))):
		image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

		yield image


def databaseFilenames(path_in, filetypeExt_in="png"):
	return sorted(glob.iglob(os.path.join(path_in, "*." + filetypeExt_in)))


def imgWindowsFromBoxes(imgPaths, imgsBoxes):
	for imgPath, boxes in zip(imgPaths, imgsBoxes):
		image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
		image = np.array(image)
		# make border for edge cases
		borderY, borderX = (128,64)
		image = cv2.copyMakeBorder(
			image, borderY, borderY, borderX, borderX, cv2.BORDER_REPLICATE)
		# for each person (box), yield the window
		for box in boxes:
			# border added to coordinates
			begX = box[0] + borderX
			begY = box[1] + borderY
			endX = box[2] + borderX
			endY = box[3] + borderY
			# make person ratio 2:1
			height = endY - begY
			width  = endX - begX
			if height/width > 2:
				# height bigger, add to width
				diff = (height - 2*width)/2
				begX -= int(diff/2)
				endX += int(diff/2)
			else:
				# width bigger, add to height
				diff = 2*width-height
				begY -= int(diff/2)
				endY += int(diff/2)
			# resize to window size exactly
			window = cv2.resize(image[begY:endY, begX:endX],(64,128))
			# cv2.imshow(imgPath, window)
			# cv2.waitKey(1000)
			yield window


def INRIAPerson(pathDB):
	"""
	pathDB: path to directory INRIAPerson/Test or Train

	return imgPathsPos, imgPathsNeg, boxesPos, windowsPos
	"""
	boxesPos = []
	imgPathsPos = list(sorted(glob.iglob(os.path.join(pathDB, "pos/*.png"))))
	imgPathsNeg = list(sorted(glob.iglob(os.path.join(pathDB, "neg/*.png"))))
	for filepath in sorted(glob.iglob(os.path.join(pathDB, "annotations/*.txt"))):
		with open(filepath, "r") as f:
			img_coord = []
			for line in f:
				if "(Xmin, Ymin)" in line:
					coord = line.split(":")[1][1:-1].replace("(", "").replace(")", "").replace(
						",", "").replace("-", " ").replace("  ", "").split(" ")
					coord = list(map(int, coord))
					# [begX, begY, endX, endY]
					img_coord.append(coord)
		boxesPos.append(np.array(img_coord))
	
	windowsPos = imgWindowsFromBoxes(imgPathsPos, boxesPos)
	## print minimum and maximum width/height
	# boxesPosFlat = np.concatenate(boxesPos)
	# width = boxesPosFlat[:, 2] - boxesPosFlat[:, 0]
	# height = boxesPosFlat[:, 3] - boxesPosFlat[:, 1]
	# print(np.amax(width))
	# print(np.amax(height))
	# print(np.amin(width))
	# print(np.amin(height))
	return imgPathsPos, imgPathsNeg, boxesPos, windowsPos
