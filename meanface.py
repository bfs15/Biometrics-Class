import yalefaces
import cv2
import os
import numpy as np
from PIL import Image

class EigenFaceRecognizer(object):
	def __init__(self):
		super(EigenFaceRecognizer, self).__init__()

	def meanface(self, images):
		mean = np.zeros(images[0].shape)

		for face in images:
			mean += face.astype(float)

		mean = mean / len(images)
		# print(mean)
		# mean = np.array(mean, 'uint8')
		# cv2.imshow("meanface >:( ", mean)
		# cv2.waitKey(250)
		return mean

	def train(self, images):
		mean = self.meanface(images)
		col_images = []
		for image in images:
			col_images.append([pixel for line in image for pixel in line])

		col_mean = [pixel for line in mean for pixel in line]

		normalized = []
		for image in col_images:
			normalized.append(np.array(image) - np.array(col_mean))

		normalized = np.array(normalized)
		l = normalized.shape[0]
		mn = normalized.shape[1]

		S = np.matrix(normalized.transpose()) * np.matrix(normalized)
		S /= len(normalized)
		print(S)
		###

		# self.evalues, self.evectors = np.linalg.eig(S)                          # eigenvectors/values of the covariance matrix
		# sort_indices = self.evalues.argsort()[::-1]                             # getting their correct order - decreasing
		# self.evalues = self.evalues[sort_indices]                               # puttin the evalues in that order
		# self.evectors = self.evectors[:,sort_indices]                             # same for the evectors

		# evalues_sum = sum(self.evalues[:])                                      # include only the first k evectors/values so
		# evalues_count = 0                                                       # that they include approx. 85% of the energy
		# evalues_energy = 0.0
		# for evalue in self.evalues:
		#     evalues_count += 1
		#     evalues_energy += evalue / evalues_sum

		#     if evalues_energy >= self.energy:
		#         break

		# self.evalues = self.evalues[0:evalues_count]                            # reduce the number of eigenvectors/values to consider
		# self.evectors = self.evectors[:,0:evalues_count]

		# #self.evectors = self.evectors.transpose()                                # change eigenvectors from rows to columns (Should not transpose) 
		# self.evectors = L * self.evectors                                       # left multiply to get the correct evectors
		# norms = np.linalg.norm(self.evectors, axis=0)                           # find the norm of each eigenvector
		# self.evectors = self.evectors / norms                                   # normalize all eigenvectors

		# self.W = self.evectors.transpose() * L                                  # computing the weights




	def predict(face):
		return 0


# Path to the Yale Dataset
path = 'yalefaces'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
print('loading yalefaces database')
images, labels = yalefaces.load(path, ["sad"], False)

recognizer = EigenFaceRecognizer()
recognizer.train(images)

def eigenFaces(self,_Ml):
	self.Ml = _Ml

	M = len(self.images)
	iH,iW = self.images[0].shape
	N2 = iH * iW

	mI = np.ravel(np.asarray(self.images)).reshape(M,-1).T #[NxM], N^2 = WxH

	N2,M = mI.shape
	mA = mI - np.average(mI,axis=1).reshape(-1,1)  #[N2xM]
	mC = np.dot(mA.T,mA) # [M,M] = [M,N2].[N2,M]

	evals, evects = npla.eig(mC) # [M,nV] -- A^T A
	evects = np.real(evects)
	eord = np.argsort(evals); eord[:] = eord[::-1] # index of sorted eigenvalues
#
	# ==> [mU] = [N2,Ml] = [N2,M].[M,Ml] 
	mU = np.dot( mA , evects[:,eord[range(self.Ml)]] )
	# normalization of eigenface to unity	
	for i in xrange(self.Ml):
		mU[:,i] = mU[:,i] / npla.norm(mU[:,i])

	return mU