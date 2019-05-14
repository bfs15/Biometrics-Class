import glob
import os

from PIL import Image
import numpy as np
from matplotlib import pylab as plt

def fingerprints(path_in, filetypeExt_in = "raw"):
	images = []
	subject_nos = []
	singular_pts = []

	pathLabel = os.path.join(os.path.dirname(path_in), "Rindex28-type")

	for pathFilename in sorted(glob.iglob(os.path.join(path_in, "*." + filetypeExt_in))):
		image = np.fromfile(pathFilename, dtype='int8')
		base = os.path.basename(pathFilename)
		name = os.path.splitext(base)[0]
		name = name.lower()
		subject_no = int(name.split('r')[0][1:])
		label = readLabel(os.path.join(pathLabel, name + ".lif"))

		image = image.reshape([300, -1])

		image = Image.fromarray(image)

		images.append(image)
		subject_nos.append(subject_no)
		singular_pts.append(label)

	return images, subject_nos, singular_pts


import json
import os


def readLabel(fileName):
	file_data = open(fileName).read()
	data = json.loads(file_data)
	# core, delta
	singular_pts = [[], []]

	# print(fileName)
	for es, s in enumerate(data["shapes"]):
		# print(s["label"].lower())
		# for p in s["points"]:
		# 	print(p)

		index = None
		if (s["label"].lower() == "core"):
			index = 0
		elif s["label"].lower() == "delta":
			index = 1

		for p in s["points"]:
			singular_pts[index].append((p[0], p[1]))
		
	print(singular_pts,'\n')
	return singular_pts

def labels(path = '.'):

	files = [os.path.join(path, f) for f in sorted(os.listdir(path))
				if f.endswith('.lif') and os.path.isfile(os.path.join(path, f))]
		
	for f in files:
		# print(f)
		readLabel(f)

