#!/usr/bin/python

# Import the required modules
import os
import sys
import getopt
import argparse
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt
import math

from irisrec import IrisRec, CASIAIris, UBIRIS


# avoiding anoying warnings
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

_waitingtime = 0.1


def main(argv):

	try:
		parser = argparse.ArgumentParser('Iris Recognition process')
		parser.add_argument('--dbname', '-d', dest='dbname',
		                    help='dataset name', default='CASIAIris')
		parser.add_argument('--dbversion', '-v', dest='dbversion',
		                    help='dataset version', default='v4-Lamp100')
		parser.add_argument('--load_eyes_path', dest='load_eyes_path')
		parser.add_argument('--save_mask_path', dest='save_mask_path')
		parser.add_argument('--load_mask_path', dest='load_mask_path')
		parser.add_argument('--save_iris_path', dest='save_iris_path')
		parser.add_argument('--load_iris_path', dest='load_iris_path')
		args = parser.parse_args()
		print args
	except getopt.GetoptError:
		print '-d <database> -n'
		print '-d CASIA  -v v4-Lamp100 -m'
		print '-d UBIRIS -v v2-40      -m'
		sys.exit(1)

	print 'loading {0}-{1} database...'.format(args.dbname, args.dbversion)
	if args.dbname == 'CASIAIris':
		db = CASIAIris(pathEye=IrisRec.DBEyePath[args.dbname][args.dbversion],
		               pathMask=IrisRec.DBMaskPath[args.dbname][args.dbversion],
		               pathIris=IrisRec.DBIrisPath[args.dbname][args.dbversion],
		               pathNorm=IrisRec.DBNormPath[args.dbname][args.dbversion])
	elif args.dbname == 'UBIRIS':
		db = UBIRIS(pathEye=IrisRec.DBEyePath[args.dbname][args.dbversion],
		            pathMask=IrisRec.DBMaskPath[args.dbname][args.dbversion])
	else:
		print 'database not load ({0}/{1})'.format(args.dbname, args.dbversion)
		sys.exit(1)

	print 'done({0})'.format(len(db.EyeImages))


if __name__ == "__main__":
	main(sys.argv[1:])
