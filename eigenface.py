
import cv2
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

import yalefaces
import ORLfaces


class EigenFaceRecognizer(BaseEstimator, ClassifierMixin):
	def __init__(self):
		super(EigenFaceRecognizer, self).__init__()

	def mean_face(self, images):
		mean_face = np.zeros(images[0].shape)

		for face in images:
			mean_face += face.astype(float)

		mean_face = mean_face / len(images)
		return mean_face

	def train(self, images, labels):
		self.labels = labels
		# Calculate mean face
		mean = self.mean_face(images)

		# Save mean face image
		# cv2.imwrite("Meanface" + ".jpg", mean)

		# Change images and mean to a single column of size m*n, instead of a matrix m x n
		col_images = []
		for image in images:
			col_images.append([pixel for line in image for pixel in line])
		
		self.mean_col_face = [pixel for line in mean for pixel in line]
		# save resolution size
		self.resolution = len(self.mean_col_face)

		# Calculate image - mean for all images
		matrixA = []
		for image in col_images:
			matrixA.append(np.array(image) - np.array(self.mean_col_face))

		# each column is a face
		# matrix of shape (len(images), resolution)
		matrixA = np.matrix(matrixA).T
		# Calculate covariance matrix
		matrixS = matrixA.T * matrixA
		matrixS /= len(matrixA)

		# calculate eigenvalues, eigenvectors
		eigenvalues, self.eigenvectors = np.linalg.eig(matrixS)
		# Get sorted indices
		indices = eigenvalues.argsort()[::-1]
		# Using indices to sort values and vectors
		eigenvalues = eigenvalues[indices]
		self.eigenvectors = self.eigenvectors[:, indices]

		# include only the most relevant eigenvectors
		eigenvalues_count = 5
		# include only the first k evectors/values so
		# that they include approx. 85% of the energy
		# eigenvalues_sum = sum(eigenvalues[:])
		# eigenvalues_count = 0
		# eigenvalues_energy = 0.0
		# energy = 0.85
		# for value in eigenvalues:
		# 	eigenvalues_count += 1
		# 	eigenvalues_energy += value / eigenvalues_sum

		# 	if eigenvalues_energy >= energy:
		# 		break

		# print('using ', eigenvalues_count, 'eigenvalues')

		eigenvalues = eigenvalues[0:eigenvalues_count]
		self.eigenvectors = self.eigenvectors[:, 0:eigenvalues_count]

		# get the true eigenvectors of matrixS (eigenfaces)
		self.eigenvectors = matrixA * self.eigenvectors
		# find the norm of each eigenvector
		norms = np.linalg.norm(self.eigenvectors, axis=0)
		# normalize each eigenvectors
		self.eigenvectors = self.eigenvectors / norms

		# Save Eigenface images
		# self.write_eigenfaces()

		# compute the projections
		self.Proj = self.eigenvectors.T * matrixA
	
	def fit(self, X, y=None):
		self.train(X, y)

	def write_eigenfaces(self):
		for i, eigenvector in enumerate(self.eigenvectors.T):
			eigenvector = np.real(eigenvector)
			eigenvector = (eigenvector - np.amin(eigenvector)) / \
	                            (np.amax(eigenvector)-np.amin(eigenvector))
			eface = np.array(255*np.real(eigenvector), dtype=np.uint8)
			eface = eface.reshape(150, 150)
			cv2.imwrite("Eigenface " + str(i) + ".jpg", eface)
			cv2.waitKey(1)

	def predict(self, face):
		# turn image matrix into a single column
		col_face = np.array(face, dtype='float64').flatten()
		# subtract mean
		col_face -= self.mean_col_face
		# transpose
		col_face = np.reshape(col_face, (self.resolution, 1))
		# project onto the Eigenspace, to find out the weights
		proj = self.eigenvectors.T * col_face

		# calculate distance to each face ||Proj - pk||
		dist = self.Proj - proj
		norms = np.linalg.norm(dist, axis=0)
		closest_face_id = np.argmin(norms)
		# return the label of the closest face
		return self.labels[closest_face_id]

	def predict_list(self, faces):
		y_pred = []
		for face in faces:
			y_pred.append(self.predict(face))
		return y_pred

	def score(self, X, y):
		y_pred = self.predict_list(X)
		return accuracy_score(y_pred, y)

if __name__ == "__main__":
	print('loading database')
	# Path to the Yale Dataset
	path = 'yalefaces'
	images, labels = yalefaces.load(path, ["sad"], False)
	# images, labels = ORLfaces.load()

	recognizer = EigenFaceRecognizer()
	recognizer.train(images, labels)

	for image, label in zip(images, labels):
		print(recognizer.predict(image), '==', label)
