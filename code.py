#!/usr/bin/python

# Import the required modules
from utils import flood_fill,bilinear_interpolation
import cv2, sys, os
import cv2.cv as cv
import math as mt
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy.linalg as npla
import scipy.misc as spm
import string

# ToDo's
## Retrieving information from mask
## Reading Masking and Normalization
## Feature generation (LBP \& Wavelet)
## Evaluation FAR vs FRR

_waitingtime = 2.0 #0.5

class IrisRec:

	DBEyePath  = dict(
      CASIAIris = {
        'v4-Lamp100'  : os.environ['HOME'] + '/databases/CASIA-IrisV4-Lamp-100/',
        'v4-Interval' : os.environ['HOME'] + '/databases/CASIA-IrisV4-Interval/'},
      UBIRIS = {
        'v2-40'       : os.environ['HOME'] + '/databases/UBIRIS.v2-40/',
        'v2-150'      : os.environ['HOME'] + '/databases/UBIRIS.v2-150'} )

	DBMaskPath  = dict(
      CASIAIris = {
        'v4-Lamp100'  : os.environ['HOME'] + '/databases/CASIA-IrisV4-Lamp-100-mask/',
        'v4-Interval' : os.environ['HOME'] + '/databases/CASIA-IrisV4-Interval-mask/'},
      UBIRIS = {
        'v2-40'       : os.environ['HOME'] + '/databases/UBIRIS.v2-40-mask/',
        'v2-150'      : os.environ['HOME'] + '/databases/UBIRIS.v2-150-mask/'} )

	DBIrisPath  = dict(
      CASIAIris = {
        'v4-Lamp100'  : os.environ['HOME'] + '/databases/CASIA-IrisV4-Lamp-100-iris/',
        'v4-Interval' : os.environ['HOME'] + '/databases/CASIA-IrisV4-Interval-iris/'},
      UBIRIS = {
        'v2-40'       : os.environ['HOME'] + '/databases/UBIRIS.v2-40-iris/',
        'v2-150'      : os.environ['HOME'] + '/databases/UBIRIS.v2-150-iris/'} )

	DBNormPath  = dict(
      CASIAIris = {
        'v4-Lamp100'  : os.environ['HOME'] + '/databases/CASIA-IrisV4-Lamp-100-norm/',
        'v4-Interval' : os.environ['HOME'] + '/databases/CASIA-IrisV4-Interval-norm/'},
      UBIRIS = {
        'v2-40'       : os.environ['HOME'] + '/databases/UBIRIS.v2-40-norm/',
        'v2-150'      : os.environ['HOME'] + '/databases/UBIRIS.v2-150-norm/'} )



	_pathEye  = '' ## virtual path
	_pathMask = ''
	_pathIris = ''
	_pathNorm = ''


	def __init__(self, pathEye = _pathEye, pathMask = _pathMask, pathIris = _pathIris, pathNorm = _pathNorm):
		self.pathEye  = pathEye
		self.pathMask = pathMask
		self.pathIris = pathIris
		self.pathNorm = pathNorm

		if self.pathEye: 
			self.getEyeImages(self.pathEye,self.pathMask,self.pathIris,self.pathNorm)
		elif self.Mask:
			self.getMaskImages(self.pathEye,self.pathMask,self.pathIris,self.pathNorm)
		elif self.pathNorm:
			self.getIrisImages(self.pathNorm)



	def Mask2Norm(self, imgIris, imgMask, wNorm):
		# angle points to normalize iris
		pts_norm = np.transpose([(mt.cos((2*mt.pi/wNorm[1])*ang), mt.sin((2*mt.pi/wNorm[1])*ang)) for ang in xrange(wNorm[1])])

		se3R = cv2.getStructuringElement(cv2.MORPH_RECT   ,(5,5)) # se 3x3 - Squar-shaped
		cy,cx = np.divide(imgIris.shape,2) ## extract info from imgMask
		radPupil  = 20
		radIris   = 40

#		imgIris = np.zeros(wNorm)
		norm_rad = []
		for i in range (wNorm[0]):
			norm_rad.append((int)(i * ((float)(radIris - radPupil)/wNorm[0]))+0.5+radPupil)

		imgNorm = np.zeros((wNorm))

		for j in range(wNorm[1]):
			for i in range(wNorm[0]):
				pt  = ( cy+pts_norm[1][j]*(norm_rad[i]) , cx+pts_norm[0][j]*(norm_rad[i]) )
				ptl = ( int(mt.floor(pt[0])), int(mt.floor(pt[1])) )

				pt1 = ( ptl[0]  , ptl[1]  , imgIris[ptl[0]  , ptl[1]  ] )
				pt2 = ( ptl[0]+1, ptl[1]  , imgIris[ptl[0]+1, ptl[1]  ] )
				pt3 = ( ptl[0]  , ptl[1]+1, imgIris[ptl[0]  , ptl[1]+1] )
				pt4 = ( ptl[0]+1, ptl[1]+1, imgIris[ptl[0]+1, ptl[1]+1] )

				# interpolate
				imgNorm[i][j] = bilinear_interpolation(pt[0], pt[1],[pt1,pt2,pt3,pt4])

		return imgNorm



class CASIAIris(IrisRec):

	_pathEye  = './databases/CASIA-IrisV4-Interval'
	_pathEye  = './databases/CASIA-IrisV4-Lamp-100'
	_pathMask = './databases/CASIA-IrisV4-Lamp-100-mask'
	_pathIris = './databases/CASIA-IrisV4-Lamp-100-iris'
	_pathNorm = './databases/CASIA-IrisV4-Lamp-100-norm'

	def getEyeImages(self,pathEye = _pathEye, pathMask = _pathMask, pathIris = _pathIris, pathNorm = _pathNorm):
		# images will contains face images
		self.EyeImages  = []
		self.MaskImages = []
		self.IrisImages = []
		self.NormImages = []

		# subjets will contains the subject identification number assigned to the image
		self.idEye = []

		if pathMask and not os.path.exists(pathMask):
			os.makedirs(pathMask)
		if pathIris and not os.path.exists(pathIris):
			os.makedirs(pathIris)
		if pathNorm and not os.path.exists(pathNorm):
			os.makedirs(pathNorm)

		subjects_paths = [os.path.join(pathEye, d) for d in os.listdir(pathEye) if os.path.isdir(os.path.join(pathEye,d))]
		for s,subject_paths in enumerate(subjects_paths, start=1):
			# Get the label of the subject
			nsb = int(os.path.split(subject_paths)[1]) 

			if pathMask and not os.path.exists(os.path.join(pathMask,os.path.split(subject_paths)[1])):
				os.makedirs(os.path.join(pathMask,os.path.split(subject_paths)[1]))
			if pathIris and not os.path.exists(os.path.join(pathIris,os.path.split(subject_paths)[1])):
				os.makedirs(os.path.join(pathIris,os.path.split(subject_paths)[1]))
			if pathNorm and not os.path.exists(os.path.join(pathNorm,os.path.split(subject_paths)[1])):
				os.makedirs(os.path.join(pathNorm,os.path.split(subject_paths)[1]))

			side_paths = [os.path.join(subject_paths, d) for d in os.listdir(subject_paths) if os.path.isdir(os.path.join(subject_paths,d))]
			for e,side_path in enumerate(side_paths, start=1):
				idEye = 2*nsb + (-1 if os.path.split(side_path)[1]  == 'L' else 0 )
				print '{0}/{1}:{2}'.format(nsb,idEye,side_path)
	
				if pathMask and not os.path.exists(os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
					os.makedirs(os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))
				if pathIris and not os.path.exists(os.path.join(pathIris,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
					os.makedirs(os.path.join(pathIris,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))
				if pathNorm and not os.path.exists(os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1])):
					os.makedirs(os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1]))

				eye_paths = [os.path.join(side_path, f) for f in os.listdir(side_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(side_path,f)) ]
				for y,eye_path in enumerate(eye_paths,start=1):
					# Read the image and convert to grayscale
					imgEye_pil = Image.open(eye_path).convert('L')
					# Convert the image format into numpy array
					imgEye = np.array(imgEye_pil, 'uint8') # normalization
					self.EyeImages.append(imgEye)
					self.idEye.append(idEye)

					print '{0}:{1}'.format(y,eye_path)
					sys.stdout.flush()
	
					imgMask, imgIris = self.SegIris(imgEye)
					if pathMask:
						imgpathMask = os.path.join(pathMask,os.path.split(subject_paths)[1],os.path.split(side_path)[1],os.path.split(eye_path)[1])
						cv2.imwrite(imgpathMask,imgMask)
					if pathIris:
						imgpathIris = os.path.join(pathIris,os.path.split(subject_paths)[1],os.path.split(side_path)[1],os.path.split(eye_path)[1])
						cv2.imwrite(imgpathIris,imgIris)

					imgNorm = self.Mask2Norm(imgIris,imgMask,(32,256))
					if pathNorm:
						imgpathNorm = os.path.join(pathNorm,os.path.split(subject_paths)[1],os.path.split(side_path)[1],os.path.split(eye_path)[1])
						cv2.imwrite(imgpathNorm,imgNorm)

#					fig, aplt = plt.subplots(2,2)
#					aplt[0,0].imshow(imgEye,cmap='Greys_r')
#					aplt[0,1].imshow(imgMask,cmap='Greys_r')
#					aplt[1,0].imshow(imgIris,cmap='Greys_r')
#					aplt[1,1].imshow(imgNorm,cmap='Greys_r')
#					plt.pause(_waitingtime)
#					plt.close()

					self.IrisImages.append(imgIris)
					self.MaskImages.append(imgMask)
					self.NormImages.append(imgNorm)
				print ' done.'


	def getIrisImages(self,pathEye = _pathEye):
		self.IrisImages = []

	def SegIris(self, imgEye):
		# angle points to find iris border
		pts_iris = np.transpose([(mt.cos((mt.pi/180)*ang), mt.sin((mt.pi/180)*ang)) for ang in xrange(360)])

		se3R = cv2.getStructuringElement(cv2.MORPH_RECT   ,(5,5)) # se 3x3 - Squar-shaped
		se3C = cv2.getStructuringElement(cv2.MORPH_CROSS  ,(5,5)) # se 3x3 - Cross-shaped
		se5R = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # se 5x5 - Rhombus-shaped

		## pupil detection
		# opening - darkening 
		ref = cv2.morphologyEx(imgEye, cv2.MORPH_OPEN, se5R, iterations=2)
#		thres_otsu,img_otsu = cv2.threshold(ref,0,255,cv2.THRESH_OTSU)
		binar = np.array(np.where(ref>30,255,0), 'uint8') # a half of otsu value

		# closing
		rec = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, se5R, iterations=3)
		edges = cv2.Canny(rec,100,120)

		#Hough circles transform
		circles = cv2.HoughCircles(edges, cv.CV_HOUGH_GRADIENT, 3.1, 100, minRadius=10, maxRadius=100)
		tt = 0
		while (circles is None and tt < 20):
			edges = cv2.dilate(edges, se5R, iterations=1)
			circles = cv2.HoughCircles(edges, cv.CV_HOUGH_GRADIENT, 3.1, 100, minRadius=10, maxRadius=100)
			tt += 1

		cy = 0
		if circles is not None:
			imgMask = np.zeros(imgEye.shape,dtype=np.uint8)

			# convert the (x, y) coordinates and radius of the circles to integers
			circles = np.round(circles[0, :]).astype("int")
			# choose the one nearest to the center of the image
			cimg = np.divide(imgEye.shape,2)
			cx,cy,radPupil = circles[np.argmin(np.sum((circles[:,0:2]-cimg)**2,1)**(0.5)),:]

			cv2.circle(imgMask, (cx, cy), radPupil, 255, 1)
#			cv2.rectangle(imgMask, (x - 5, y - 5), (x + 5, y + 5), 128, -1)
		else:
			print 'Pupil not detected'
			sys.exit(1)

#		fig, aplt = plt.subplots(1,2)
#		aplt[0].imshow(imgEye,cmap='Greys_r')
#		aplt[1].imshow(imgMask,cmap='Greys_r')
#		aplt[1,0].imshow(imgIris,cmap='Greys_r')
#		aplt[1,1].imshow(imgNorm,cmap='Greys_r')
#		plt.pause(0.1)
#		plt.close()

		if len(circles) > 1:
			print circles

		## iris contour detection
		if radPupil is not None and len(circles) >= 1:
			# hist equalized
			image_eq = cv2.equalizeHist(imgEye)

			#calc intensity
			itss = []
			# initial distance
			in_dist = 20

			im_h, im_w = np.shape(imgEye)

			# max radius to concentric circles
			maxr = min(im_h - cy, im_w - cx, cx, cy)-radPupil-in_dist

			if maxr > 0:
				yim = (cy + pts_iris[1]*(radPupil+in_dist)).astype(int)
				xim = (cx + pts_iris[0]*(radPupil+in_dist)).astype(int)

				# computing concentric circles
				its = image_eq[yim, xim]
				for rad in range(1,maxr,1):
					its_aux = image_eq[((cy + pts_iris[1]*(radPupil+in_dist+rad)).astype(int), 
                                        (cx + pts_iris[0]*(radPupil+in_dist+rad)).astype(int))]
					itss.append(np.mean(np.abs(its - its_aux)))
					its = its_aux

			radIris = radPupil+in_dist+range(1,maxr,1)[itss.index(max(itss))]
			cv2.circle(imgMask, (cx, cy), radIris, 255, 1)

			h,w = imgMask.shape[:2]
			mask = np.zeros((h+2,w+2),np.uint8)
			cv2.floodFill(imgMask, mask, (cx+radPupil+1, cy), 255);

			imgIris = imgEye.copy()
			imgIris[np.where(imgMask!=255)] = 0
		else:
			print '{0}.circles detected'.format(len(circles))
			sys.exit(1)

		return imgMask,imgIris
	



class UBIRIS(IrisRec):

	_pathEye  = './UBIRIS.v2-40'
	_pathMask = './UBIRIS.v2-40'

	def getEyeImages(self,pathEye = _pathEye, pathMask = _pathMask):
		# images will contains face images
		self.EyeImages = []
		if pathMask:
			self.MaskImages = []
		# subjets will contains the subject identification number assigned to the image
		self.idEye = []

		imgEye_paths = [os.path.join(pathEye, f) for f in os.listdir(pathEye) if f.startswith('C') and f.endswith('.tiff') and os.path.isfile(os.path.join(pathEye,f))]

		for i,imgEye_path in enumerate(imgEye_paths, start=1):
			# Parse image name to get idEye,idSession,idImg
			imgEye_name = os.path.split(imgEye_path)[1].split('.')[0].translate(string.maketrans("CSI","   "))
			(idEye,idSes,idImg) = [int(id) for id in imgEye_name.split('_')]

			# Read and the image and convert to grayscale and into numpy array
			image_pil = Image.open(imgEye_path).convert('L')
			image = np.array(image_pil, 'uint8') # normalization
			self.EyeImages.append(image)
			self.idEye.append((idEye))

			if pathMask:
				imgMask_name = 'OperatorA_C{0}_S{1}_I{2}.tiff'.format(idEye,idSes,idImg)
				imgMask_path = os.path.join(pathMask,imgMask_name)
				imageMask_pil = Image.open(imgMask_path).convert('L')
				imageMask = np.array(imageMask_pil, 'uint8') # normalization
				self.MaskImages.append(imageMask)


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