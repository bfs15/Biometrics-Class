
from PIL import Image
import numpy as np
from scipy import ndimage, misc, signal
import scipy

def contrast(image, alpha = 150, y = 95):
   mean = np.mean(image)
   var = np.std(image)
   image = alpha + y * (image - mean) / var
   image = image + 255 - image.max()
   return image


def median_filter(data, filter_size):
   return ndimage.median_filter(data, filter_size)


def gradient(img, blk_sz):
   # dx = ndimage.sobel(img, 0)  # horizontal derivative
   # dy = ndimage.sobel(img, 1)  # vertical derivative
   # scipy.misc.imsave('sobel_dx.jpg', dx)
   # scipy.misc.imsave('sobel_dy.jpg', dy)

   dx = sobel_filter(img, 0)
   dy = sobel_filter(img, 1)
   # scipy.misc.imsave('sobel_dy2.jpg', dx)
   # scipy.misc.imsave('sobel_dx2.jpg', dy)

   img_alpha_x = dx**2 - dy**2
   img_alpha_y = 2 * dx * dy

   img_alpha_x_block = [[np.sum(img_alpha_x[index_y*blk_sz: index_y*blk_sz + blk_sz, index_x*blk_sz: index_x*blk_sz + blk_sz]) / blk_sz**2
                  for index_x in range(img.shape[0]//blk_sz)]
                 for index_y in range(img.shape[1]//blk_sz)]
   
   img_alpha_y_block = [[np.sum(img_alpha_y[index_y*blk_sz: index_y*blk_sz + blk_sz, index_x*blk_sz: index_x*blk_sz + blk_sz]) / blk_sz**2
                  for index_x in range(img.shape[0]//blk_sz)]
                 for index_y in range(img.shape[1]//blk_sz)]


   img_alpha_x_block =  np.array(img_alpha_x_block)
   img_alpha_y_block =  np.array(img_alpha_y_block)

   orientation_block = np.arctan2(img_alpha_y_block, img_alpha_x_block) / 2

   return orientation_block


def sobel_filter(img, axis):
   img = img.astype(np.float)
   
   if axis == 0:
      # x axis
      # wikipedia
      # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
      # scipy
      kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float)
   elif axis == 1:
      # y axis
      # wikipedia
      # kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float)
      # scipy
      kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)

   convolved = signal.convolve2d(
       img, kernel, mode='same', boundary='wrap', fillvalue=0)

   return convolved
