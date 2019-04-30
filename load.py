import glob
import os

from PIL import Image
import numpy as np
from matplotlib import pylab as plt

def fingerprints(path_in, filetypeExt_in = "raw"):
   images = []
   for pathFilename in sorted(glob.iglob(os.path.join(path_in, "*." + filetypeExt_in))):
      image = np.fromfile(pathFilename, dtype='int8')
      image = image.reshape([300, -1])

      # plt.imshow(Image.fromarray(image))
      # plt.show()

      image = Image.fromarray(image)

      images.append(image)

   return images
