#!/usr/bin/python

# Import the required modules
import sys
import getopt
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
import math

from faceid import FaceId
from faceid import YaleFaces
from faceid import ORLFaces

# avoiding annoying warnings
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

_waitingtime = 0.1


def main(argv):

	try:
		opts, args = getopt.getopt(argv, "d:n:mfer")
	except getopt.GetoptError:
		print('-d <database> -f -m -r -e -n <#EF>')
		sys.exit(1)

	# default
	dbn = 'yale'
	bMf = False
	bEf = False
	Ml = 0
	bRec = False
	bFaceDet = False

	for opt, arg in opts:
		if opt == '-d':
			dbn = arg
		if opt == '-n':
			Ml = int(arg)
		if opt == '-f':
			bFaceDet = True
		if opt == '-m':
			bMf = True
		if opt == '-e':
			bEf = True
		if opt == '-r':
			bRec = True

	print('loading {0} database...'.format(dbn))
	if dbn == 'yale':
		db = YaleFaces(FaceId.DBPath[dbn], _bFaceDetect=bFaceDet)
	elif dbn == 'orl':
		db = ORLFaces(FaceId.DBPath[dbn], _bFaceDetect=bFaceDet)

#	else:
#		error
	print('done({0})'.format(len(db.images)))

	# meanFace?
	if bMf:
		print('computing meanFace1')
		mface = db.meanFace()
		fig = plt.figure()
		imageplt = plt.imshow(mface, cmap='Greys_r')
		plt.pause(_waitingtime)

	# reconstructFace
	if bRec:
		print('reconstruction')

		mface = db.meanFace()
		mfacel = np.array(np.ravel(mface), dtype=np.float64).reshape(-1, 1)
		efaces = db.eigenFaces(Ml)  # [KxMl]

		ims1 = None  # for exhibition
		for img in db.images:
			imgl = np.array(np.ravel(img), dtype=np.float64).reshape(-1, 1)
			pface = np.dot(efaces.T, (imgl - mfacel))  # [Mlx1] = [MlxK].[Kx1]
			rface = mfacel + np.dot(efaces, pface)      # [Kx1] =  [KxMl].[Mlx1]
			# visualization
			rface = (rface - np.amin(rface)) / (np.amax(rface)-np.amin(rface))
			rface = np.array(255*rface, dtype=np.uint8)
			rface = np.array(rface, dtype=np.uint8)
			rface = rface.T.reshape(img.shape)

			mse = np.mean((img.astype(float) - rface.astype(float)) ** 2)
			if mse == 0:
				psnr = 100
			else:
				psnr = 20 * math.log10(255/math.sqrt(mse))
			print('psnr: {0}, mse: {1}'.format(psnr, mse))

			if ims1 is None:
				ims1 = plt.subplot(1, 2, 1).imshow(img, cmap='Greys_r')
				ims2 = plt.subplot(1, 2, 2).imshow(rface, cmap='Greys_r')
			else:
				ims1.set_data(img)
				ims2.set_data(rface)
			plt.pause(_waitingtime)

	# eigenfaces
	if bEf:
		print('computing eigenFaces ({0})'.format(Ml))
		efaces = db.eigenFaces(Ml)
		efacesV = db.eigenFaces2Img(efaces)

		fig = plt.figure()
		ims = None  # for exhibition
		for eim in efacesV:
			if ims is None:
				ims = plt.imshow(eim, cmap='Greys_r')
			else:
				ims.set_data(eim)
			plt.pause(_waitingtime)


if __name__ == "__main__":
	main(sys.argv[1:])
