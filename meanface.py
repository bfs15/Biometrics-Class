
import cv2
import os
import numpy as np
from PIL import Image

import yalefaces
import ORLfaces

class EigenFaceRecognizer(object):
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
		evalues_count = 5
		eigenvalues = eigenvalues[0:evalues_count]
		self.eigenvectors = self.eigenvectors[:, 0:evalues_count]

		# get the true eigenvectors of matrixS (eigenfaces)
		self.eigenvectors = matrixA * self.eigenvectors
		# find the norm of each eigenvector
		norms = np.linalg.norm(self.eigenvectors, axis=0)
		# normalize each eigenvectors
		self.eigenvectors = self.eigenvectors / norms

		# computing the weights
		self.W = self.eigenvectors.transpose() * matrixA

	def predict(self, face):
		# turn image matrix into a single column
		col_face = np.array(face, dtype='float64').flatten()
		# subtract mean
		col_face -= self.mean_col_face
		# transpose
		col_face = np.reshape(col_face, (self.resolution, 1))
		# project onto the Eigenspace, to find out the weights
		proj = self.eigenvectors.T * col_face

		# calculate distance to each face ||w - pk||
		dist = self.W - proj
		norms = np.linalg.norm(dist, axis=0)
		closest_face_id = np.argmin(norms)
		# return the faceid (1..40)
		return labels[closest_face_id]


# Path to the Yale Dataset
path = 'yalefaces'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
print('loading yalefaces database')
# images, labels = yalefaces.load(path, ["sad"], False)
images, labels = ORLfaces.load()

recognizer = EigenFaceRecognizer()
recognizer.train(images, labels)

print(recognizer.predict(images[25]))
