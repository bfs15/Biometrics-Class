
from PIL import Image
import numpy as np
from scipy import ndimage, misc
# import scipy

def contrast(image, alpha = 150, y = 95):
   mean = np.mean(image)
   var = np.std(image)
   image = alpha + y * (image - mean) / var
   image = image + 255 - image.max()
   return image


def median_filter(data, filter_size):
   return ndimage.median_filter(data, filter_size)
