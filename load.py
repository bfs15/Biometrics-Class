import glob
import os

from PIL import Image
import cv2
import numpy as np
from matplotlib import pylab as plt

import json
import os

def database(path_in, filetypeExt_in="png"):
	images = []
	image_arg = 0
	for pathFilename in sorted(glob.iglob(os.path.join(path_in, "*." + filetypeExt_in))):
		image = cv2.imread(pathFilename,cv2.IMREAD_GRAYSCALE)
		# image = Image.open(pathFilename)

		yield image, image_arg

		image_arg += 1

