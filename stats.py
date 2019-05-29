#!/usr/bin/python

import compare

# Import the required modules
import cv2
import sys
import os
# import cv2.cv as cv
import math as mt
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
plt.style.use('dark_background')


from PIL import Image
import numpy as np
import numpy.linalg as npla
import scipy.misc as spm
import string
import time
# Wavelet
import pywt
# LBP
# import skimage
from skimage import feature

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score


## Verification
def DETCurve(fps, fns):
	"""
	Given false positive and false negative rates, produce a DET Curve.
	The false positive rate is assumed to be increasing while the false
	negative rate is assumed to be decreasing.
	"""
	axis_min = min(fps[0], fns[-1])
	fig, ax = plt.subplots()
	plt.xlabel("FAR")
	plt.ylabel("FRR")
	plt.plot(fps, fns, '-|')
	plt.yscale('log')
	plt.xscale('log')
	ax.get_xaxis().set_major_formatter(
			FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
	ax.get_yaxis().set_major_formatter(
			FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
	ticks_to_use = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.25]
	ax.set_xticks(ticks_to_use)
	ax.set_yticks(ticks_to_use)
	plt.axis([0.01, 1, 0.01, 1])
	# plt.show()


def verificationTestNP(distances, labels, threshold):
	falseTestsNo = 0
	positiveTestsNo = 0
	FAR = 0
	FRR = 0
	labels = np.array(labels)

	for i in range(len(labels)):
		# extract distances between different subjects
		indexes = np.where(labels != labels[i])
		falseTests = distances[i][indexes]
		# calculate how many are accepted when they should have been rejected
		FAR += (falseTests <= threshold).sum()
		# update total number of tests
		falseTestsNo += len(falseTests)

		# extract distances between same subject
		indexes = np.where(labels == labels[i])[0]
		# remove distance between same sample
		index = np.argwhere(indexes == i)
		indexes = np.delete(indexes, index)
		# extract distances between different subjects
		positiveTests = distances[i][indexes]
		# calculate how many are accepted when they should have been rejected
		FRR += (positiveTests > threshold).sum()
		positiveTestsNo += len(positiveTests)
		# print("indexes", indexes)
	return FAR/float(falseTestsNo), FRR/float(positiveTestsNo)


def curveFARFRR(distances, labels):
	fpsList = []
	fnsList = []
	distances = np.array(distances)
	for threshold in np.linspace(0.0, 1.0, num=int(1.0/0.01)):
		FAR, FRR = verificationTestNP(distances, labels, threshold)
		fpsList.append(FAR)
		fnsList.append(FRR)
	return fpsList, fnsList


# input data and distance function such that
# distanceFunction(data[i], data[j]) returns the distance between elements i and j
# distance is an array of metrics in which to measure distance
# [match_score, mse, ...] # or only one e.g. [match_score]
# return
# a list of distance matrixes
# each matrix being about a different metric
# return[i] is the distance matrix of metric i
# i is indexed in array returned by distanceFunction = [match_score, mse, ...]
def distanceMatrix(data, distanceFunction):
	print("Computing distance matrix")
	sys.stdout.flush()
	# initialize list
	distances = [None]*len(data)
	for i in range(len(data)):
		# initialize list
		distances[i] = [None]*len(data)
		for j in range(len(data)):
			distances[i][j] = distanceFunction(data[i], data[j])
	
	distances = np.array(distances)
	# split_distances is
	# a list of distance matrixes
	# each matrix being about a different metric
	# distance is an array of multiple score calculations [mse, match_score, ...]
	# len(distances.shape[-1]) == len(distanceFunction(data[i], data[j]))
	# distances.shape == (len(data), len(data), number_of_scores)
	# split by the number of metrics there are
	split_distances = np.split(distances, distances.shape[-1], axis=2)
	# each element of split_distances is now an array of size 1
	# squeeze [[0], [1], [2], ...] => [0, 1, 2, ...]
	split_distances = np.squeeze(split_distances)
	print(split_distances)
	sys.stdout.flush()
	return split_distances


def identification(distances, labels):
	accuracies = []
	distances[0] = 1 - distances[0]
	image_i = 0
	distances_i_to_n = np.ma.array(distances[0][0], mask=False)
	distances_i_to_n.mask[image_i] = True
	closest_i = np.argmin(distances_i_to_n)
	
	for metric_i in range(distances.shape[0]):
		metric_distance_matrix = distances[metric_i]
		# normalize matrix
		x = metric_distance_matrix
		metric_distance_matrix = (x-np.min(x))/(np.max(x)-np.min(x))

		correct_no = 0

		for image_i in range(metric_distance_matrix.shape[0]):
			distances_i_to_n = np.ma.array(metric_distance_matrix[image_i], mask=False)
			distances_i_to_n.mask[image_i] = True

			closest_i = np.argmin(distances_i_to_n)

			if(labels[closest_i] == labels[image_i]):
				correct_no += 1
			
		accuracies.append(correct_no / metric_distance_matrix.shape[0])

	return accuracies


if __name__ == "__main__":
	## Iris mask IoU evaluation
	IoUList = np.array(IoUList)
	print('IoU mean:', np.mean(IoUList))
	print('IoU var:', np.var(IoUList))

	## Computing distance array

	distances = distanceMatrix(data, distanceFunction)

	## Computing FAR, FRR

	print("Computing FAR, FRR")
	sys.stdout.flush()

	for metric_i in len(distances):
		metric_distance_matrix = distances[metric_i]
		fpsList, fnsList = curveFARFRR(metric_distance_matrix, labels)

		DETCurve(fpsList, fnsList)
		plt.savefig('DETCurve.png')

		diff = np.abs(np.array(fpsList) - np.array(fnsList))

		argEER = np.argmin(diff)
		EER = np.min(diff)

		print("argEER", argEER)
		print("EER", EER)

		print("fps", fpsList[argEER])
		print("fns", fnsList[argEER])

	## Identification

	# test data with SVM
	model = LinearSVC(C=100.0, random_state=42)
	k_fold = None
	max_fold = 40
	n_splits = 40
	for i in range(max_fold):
		try:
			n_splits = max_fold-i
			k_fold = StratifiedKFold(n_splits=n_splits)
			print("n_splits: \t", n_splits)
			print(cross_val_score(model, dataLBP, labels, cv=k_fold, n_jobs=-1))
			print(cross_val_score(model, data, labels, cv=k_fold, n_jobs=-1))
			break
		except ValueError as error:
			pass

	def testIdentification(X_train, X_test, y_train, y_test):
		model.fit(X_train, y_train)
		correct = 0
		for i, data in enumerate(X_test):
			prediction = model.predict(data)
			if y_test[i] == prediction:
				correct += 1
		accuracy = correct / float(len(X_test))
		return accuracy
	accuracyList = []
	k_fold = StratifiedKFold(n_splits=4)
	X = np.array(data)
	y = np.array(labels)
	k_fold.get_n_splits(X)
	for train_index, test_index in k_fold.split(X,labels):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		accuracy = testIdentification(X_train, X_test, y_train, y_test)
		accuracyList.append(accuracy)
	accuracyList = np.array(accuracyList)
	print('Identification mean:', np.mean(accuracyList))
	print('Identification var:', np.var(accuracyList))

