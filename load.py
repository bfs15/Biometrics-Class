
import glob
import os

import cv2

def database(path_in, filetypeExt_in="png"):
    images = []
    
    for pathFilename in sorted(glob.iglob(os.path.join(path_in, "*." + filetypeExt_in))):
        image = cv2.imread(pathFilename, cv2.IMREAD_GRAYSCALE)

        yield image