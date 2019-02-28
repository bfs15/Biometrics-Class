#!/usr/bin/python

# Import the required modules
import cv2, os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy.linalg as npla
import scipy.misc as spm
	

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

class FaceId:

	DBPath = dict(yale = os.environ['HOME'] + '/databases/yalefaces/',
	              orl  = os.environ['HOME'] + '/databases/orl_faces/')

	Ml = 0


	_path = '' ## virtual path

	def __init__(self, path = _path, _bFaceDetect = False, _sz = (150,150) ):
		self.path = path
		self.bFaceDetect = _bFaceDetect
		self.szFaceDetect = _sz
		self.get_images_and_labels(self.path)


	def meanFace(self):
		imgmf = np.zeros(self.images[0].shape, dtype=np.uint32) # due to integer summations uint32
		for im in self.images:
			imgmf = imgmf + im
		imgmf = imgmf / len(self.images)

		return imgmf


	def meanFace2(self):
		mMF = np.average(np.array(self.images),axis=0) # computes the average face

		return mMF


	def eigenFaces(self,_Ml):
		self.Ml = _Ml

		M = len(self.images); iH,iW = self.images[0].shape;	N2 = iH * iW

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


	def eigenFaces2Img(self,efaces):
		M = len(self.images)
		iH, iW = self.images[0].shape
		N2 = iH * iW
		print 'M: {0}, iH: {1}, iW: {2}, N2: {3}'.format(M, iH, iW, N2)
		efaces = (efaces - np.amin(efaces)) / (np.amax(efaces)-np.amin(efaces))
		efaces = np.array(255*efaces,dtype=np.uint8)
		efaces = efaces.T.reshape(self.Ml,iH,iW)

		return efaces


	def projectedFace(self,img,mface,efaces):
		pfaces = np.dot( efaces.T, (np.ravel(img).reshape(-1,1) - np.ravel(mface).reshape(-1,1)) )

		return pfaces



class ORLFaces(FaceId):

	_path = './orl_faces'

	def get_images_and_labels(self,path = _path):
		# images will contains face images
		self.images = []
		# subjets will contains the subject identification number assigned to the image
		self.subjects = []

		if self.bFaceDetect:
			print 'Detecting faces...'

		subjects_paths = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
		for s,subject_paths in enumerate(subjects_paths, start=1):

			# Get the label of the image
			nbr = int(os.path.split(subject_paths)[1].split(".")[0].replace("s",""))
#			print 'sub: {0}--{1}'.format(subject_paths,nbr)
	
			subject_path = [os.path.join(subject_paths, f) for f in os.listdir(subject_paths) if f.endswith('.pgm') and os.path.isfile(os.path.join(subject_paths,f)) ]

			for image_path in subject_path:
#				print 'img: {0}'.format(image_path)
				# Read the image and convert to grayscale
				image_pil = Image.open(image_path).convert('L')
				# Convert the image format into numpy array
				image = np.array(image_pil, 'uint8') # normalization
				if self.bFaceDetect:
					faces = faceCascade.detectMultiScale(image)
					for (x,y,w,h) in faces:
						self.images.append(spm.imresize(image[y:y+h,x:x+w],self.szFaceDetect))
						self.subjects.append(nbr)
				else:
					self.images.append(image)
					self.subjects.append(nbr)


	
#			print 'sub: {0}({1}#) - {2}'.format(s,len(subject_path),subject_paths)


class YaleFaces(FaceId):

	_path = './yalefaces'
	# classes: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.
	class_labels = ['.centerlight', '.glasses', '.happy', '.leftlight', '.noglasses', '.normal', '.rightlight', '.sad', '.sleepy', '.surprised','.wink']
	# Note that the image "subject04.sad" has been corrupted and has been substituted by "subject04.normal".
	# Note that the image "subject01.gif" corresponds to "subject01.centerlight" :~ mv subject01.gif subject01.centerlight


	def get_images_and_labels(self,path = _path):
	    # Append all the absolute image paths in a list image_paths
		# We will not read the image with the .sad extension in the training set
		# Rather, we will use them to test our accuracy of the training

		# images will contains face images
		self.images = []
		# subjets will contains the subject identification number assigned to the image
		self.subjects = []
		# classes
		self.classes = []

		if self.bFaceDetect:
			print 'Detecting faces...'

		for c,class_label in enumerate(self.class_labels,start=1):
			image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(class_label)]

			for image_path in image_paths:
#				print 'Image: ' + image_path
				# Read the image and convert to grayscale
				image_pil = Image.open(image_path).convert('L')
				# Convert the image format into numpy array
				image = np.array(image_pil, 'uint8')  # normalization
				# Get the label of the image
				nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

				if self.bFaceDetect:
					faces = faceCascade.detectMultiScale(image)
					for (x,y,w,h) in faces:
						self.images.append(spm.imresize(image[y:y+h,x:x+w],self.szFaceDetect))
						self.subjects.append(nbr)
						self.classes.append(class_label)
				else:
					self.images.append(image)
					self.subjects.append(nbr)
					self.classes.append(class_label)

#			print 'class_label: {0}({1}#) - {2}'.format(c,len(image_paths), class_label)

## Path to the Yale Dataset
#path = '/home/menotti/databases/yalefaces/'
#print 'loading Yalefaces database'
#yale = YaleFaces(path)
#yale.eigenFaces2()

## Path to the ORl Dataset
#path = '/home/menotti/databases/orl_faces/'
#print 'loading ORL database'
#orl = ORL(path)
#orl.eigenFaces2()

