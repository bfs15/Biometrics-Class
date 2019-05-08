
from PIL import Image
import numpy as np
from scipy import ndimage, misc, signal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm
import cv2
import math
import sys


def normalize(image):
   image = np.float64(image)
   image = (image - image.min()) / (image.max() - image.min())
   image = image * 255
   return np.uint8(image)

def contrast(image, alpha=150, y=95):
# def contrast(image, alpha=150, y=95):
   image = np.float64(image)

   mean = np.mean(image)
   var = np.std(image)
   image = alpha + y * (image - mean) / var

   image = np.where(image<0, 0,image)
   image = normalize(image)
   return np.uint8(image)


def median_filter(img, filter_size):
   median = img.copy()
   hs = filter_size//2
   cv2.erode
   for i in range(hs, img.shape[0]-hs):
      for j in range(hs, img.shape[1]-hs):
         pixels = sorted(img[i-hs: i+hs+1, j-hs: j+hs+1].flatten())
         # dislocated median
         median[i, j] = pixels[8+ filter_size*filter_size // 2]
   return median
   # return cv2.medianBlur(img, filter_size)

def binarize(img, blk_sz):
   img = img.copy()
   per25 = np.percentile(img, 25)
   print('per25')
   print(per25)
   per50 = np.percentile(img, 50)
   print('per50')
   print(per50)

   # hist, bins = np.histogram(img, 256, [0, 256])
   # plt.hist(img.ravel(), 256, [0, 256])
   # plt.title('Histogram for gray scale picture')
   # plt.show()
   means = np.zeros(img.shape)

   # number of blocks in a dimension
   blk_no_y, blk_no_x = (int(img.shape[0]//blk_sz)+1, int(img.shape[1]//blk_sz)+1)
   blk_mean = np.zeros((blk_no_y, blk_no_x))
   # for each block i,j
   for i in range(blk_no_y):
      for j in range(blk_no_x):
            block = img[blk_sz*i: blk_sz*(i+1), blk_sz*j: blk_sz*(j+1)]
            blk_mean[i, j] = np.mean(block)

   img = np.where(img < per25, 0, img)
   img = np.where(img > per50, 255, img)

   for i in range(1, img.shape[0]-1):
      for j in range(1, img.shape[1]-1):
         if(img[i, j] == 0 or img[i, j] == 255):
            continue
         block = img[i-1: i+1, j-1: j+1]
         block = np.ma.array(block.flatten(), mask=False)
         block.mask[len(block)//2] = True
         if(block.mean() >= blk_mean[i//blk_sz, j//blk_sz]):
            img[i, j] = 255
         else:
            img[i, j] = 0
   return img


def region_of_interest(img, blk_sz, gray_out=125):
   img = img.copy()
   # number of blocks in a dimension
   blk_no_y, blk_no_x = (int(img.shape[0]//blk_sz), int(img.shape[1]//blk_sz))
   blk_mean = np.zeros((blk_no_y, blk_no_x))
   blk_std = np.zeros((blk_no_y, blk_no_x))
   # for each block i,j
   for i in range(blk_no_y):
      for j in range(blk_no_x):
            block = img[blk_sz*i: blk_sz*(i+1), blk_sz*j: blk_sz*(j+1)]
            blk_mean[i, j] = np.mean(block)
            blk_std[i, j] = np.std(block)
   # normalize values
   blk_mean = (blk_mean-np.min(blk_mean))/(np.max(blk_mean) - np.min(blk_mean))
   blk_std = (blk_std-np.min(blk_std))/(np.max(blk_std) - np.min(blk_std))
   # constant
   max_dist_to_center = np.sqrt(
       np.square((blk_no_y/2)) + np.square((blk_no_x/2)))
   # max_dist_to_center = (blk_no_y + blk_no_x)/2
   # for each block i,j
   for i in range(blk_no_y):
      for j in range(blk_no_x):
            w0 = 0.5
            w1 = 0.5
            dist_to_center = np.sqrt(
                np.square(i-(blk_no_y/2)) + np.square(j-(blk_no_x/2)))
            w2 = 1 - (dist_to_center/max_dist_to_center)

            v = w0*(1-blk_mean[i, j]) + w1*blk_std[i, j] + w2
            if (v < 0.8):
               # gray out block
               img[blk_sz*i:blk_sz*(i+1), blk_sz*j:blk_sz*(j+1)] = gray_out

   return img
