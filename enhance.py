
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
         median[i, j] = pixels[8 + filter_size*filter_size // 2]
   return median
   # return cv2.medianBlur(img, filter_size)

def draw_orientation_map(img, angles, block_size, thicc=1):
   '''
      img - Image array.
      angles - orientation angle at (x,y) pixel.
      block_size - block size of the orientation field.
      thicc - Thickness of the lines to plot.
   '''
   x_center = y_center = float(block_size)/2.0  # y for rowq and x for columns
   blk_no_y, blk_no_x = angles.shape
   # add color channels to ndarray
   img_draw = np.stack((img,)*3, axis=-1)
   for i in range(0, blk_no_y):
      for j in range(0, blk_no_x):
         # point is (x, y) # y top of the image is maximum y, in the loop is 0, therefore (blk_no_y-1-i)
         y, x = ((i)*block_size + y_center,
                 (j)*block_size + x_center)
         angle = angles[i][j]
         length = block_size # total length of the line
         # down, left # add on y, sub on x
         starty = int(y + np.sin(angle)*length/2)
         startx = int(x - np.cos(angle)*length/2)
         # find the end point
         # up, right # sub on y, add on x
         endy = int(y - np.sin(angle)*length/2)
         endx = int(x + np.cos(angle)*length/2)
         cv2.line(img_draw, (startx, starty), (endx, endy), (255, 0, 0), thicc)

   return img_draw

def draw_singular_points(image, poincare, block_sz, tolerance=2, thicc=2):
   '''
      image - Image array.
      poincare - Poincare index matrix of each block.
      block_sz - block size of the orientation field.
      tolerance - Angle tolerance in degrees.
      thicc - Thickness of the lines to plot.
   '''
   # add color channels to ndarray
   image_color = np.stack((image,)*3, axis=-1)
   for i in range(1, image.shape[0]//block_sz - 1):
      for j in range(1, image.shape[1]//block_sz - 1):
            # if adjacent to blocks not in roi, ignore
            if(image[(i)*block_sz, (j-1)*block_sz] == 125 or image[(i)*block_sz, (j+1)*block_sz] == 125 or image[(i-1)*block_sz, (j)*block_sz] == 125 or image[(i+1)*block_sz, (j)*block_sz] == 125):
               continue
            angle_core = 180
            color = matplotlib.cm.hot(abs(poincare[i, j]/360))
            color = (color[0]*255, color[1]*255, color[2]*255)
            if (np.isclose(poincare[i, j], angle_core, 0, tolerance)):
               cv2.circle(image_color, (int((j+0.5)*block_sz), int((i+0.5) * block_sz)),
                           int(block_sz/2), color, thicc)
            angle_delta = -180
            # if (poincare[i, j] > angle_delta-tolerance and poincare[i, j] < angle_delta+tolerance):
            if (np.isclose(poincare[i, j], angle_delta, 0, tolerance)):
               cv2.rectangle(image_color, (j*block_sz, i*block_sz),
                             ((j+1)*block_sz, (i+1)*block_sz), (255, 125, 0), thicc)
            whorl = 360
            # if (poincare[i, j] > whorl-tolerance and poincare[i, j] < whorl+tolerance):
            if (np.isclose(poincare[i, j], whorl, 0, tolerance)):
               cv2.circle(image_color, (int((j+0.5)*block_sz), int((i+0.5) * block_sz)),
                          int(block_sz/2), (0, 200, 200), thicc)
   return image_color

def gradient(img, blk_sz):
   # dx = sobel_filter(img, 0)
   # dy = sobel_filter(img, 1)

   dy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
   dx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)


   img_alpha_x = dx*dx - dy*dy
   img_alpha_y = 2 * np.multiply(dx, dy)
   
   img_alpha_x_block = [[np.sum(img_alpha_x[index_y: index_y + blk_sz, index_x: index_x + blk_sz]) / blk_sz**2
                  for index_x in range(0, img.shape[0], blk_sz)]
                 for index_y in range(0, img.shape[1], blk_sz)]
   
   img_alpha_y_block = [[np.sum(img_alpha_y[index_y: index_y + blk_sz, index_x: index_x + blk_sz]) / blk_sz**2
                         for index_x in range(0, img.shape[0], blk_sz)]
                 for index_y in range(0, img.shape[1], blk_sz)]

   img_alpha_x_block = np.array(img_alpha_x_block)
   img_alpha_y_block = np.array(img_alpha_y_block)

   orientation_blocks = np.arctan2(img_alpha_y_block, img_alpha_x_block) / 2

   return orientation_blocks


def smooth_gradient(orientation_blocks, blk_sz):
   orientation_blocks_smooth = np.zeros(orientation_blocks.shape)
   blk_no_y, blk_no_x = orientation_blocks.shape
   # Consistency level, filter of size (2*cons_lvl + 1) x (2*cons_lvl + 1)
   cons_lvl = 1
   for i in range(cons_lvl, blk_no_y-cons_lvl):
      for j in range(cons_lvl, blk_no_x-cons_lvl):
         area_sin = area_cos = orientation_blocks[i-cons_lvl: i +
                                   cons_lvl, j-cons_lvl: j+cons_lvl]
                                   
         mean_angle_sin = np.sum(np.sin(2*area_sin))
         mean_angle_cos = np.sum(np.cos(2*area_cos))
         mean_angle = np.arctan2(mean_angle_sin, mean_angle_cos) / 2
         orientation_blocks_smooth[i, j] = mean_angle

   return orientation_blocks_smooth


def poincare_index(orientation_blocks, blk_sz):
   poincare = np.zeros(orientation_blocks.shape)
   blk_no_y, blk_no_x = orientation_blocks.shape
   filter_sz = 1
   for i in range(filter_sz, blk_no_y-filter_sz):
      for j in range(filter_sz, blk_no_x-filter_sz):
         index = 0
         cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

         def get_angle(left, right):
            angle = left - right
            if abs(angle) > 180:
               angle = -1 * np.sign(angle) * (360 - abs(angle))
            return angle

         deg_angles = [np.degrees(orientation_blocks[i - k][j - l]) % 180 for k, l in cells]
         # len(cells)-1 == 8
         for k in range(len(cells)-1):
            if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
                  deg_angles[k + 1] += 180
            index += get_angle(deg_angles[k], deg_angles[k + 1])

         poincare[i, j] = index
         # # if(index > 170 and index < 190):
         # # cells = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
         # deg_angles = [np.degrees(orientation_blocks[i + k][j + l]) % 180 for k, l in cells]
         # # print(deg_angles)
         # index = 0
         # for k in range(len(cells)-1):
         #    # diff = get_angle(deg_angles[k], deg_angles[k + 1])
         #    # diff = deg_angles[k] - deg_angles[k + 1]
         #    # if(get_angle(deg_angles[k], deg_angles[k + 1]) != (deg_angles[k] - deg_angles[k + 1])):
         #    #    print(get_angle(deg_angles[k], deg_angles[k + 1])," !=== " ,(deg_angles[k] - deg_angles[k + 1]))
         #    if (abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90):
         #       deg_angles[k + 1] += 180
         #    index += get_angle(deg_angles[k], deg_angles[k + 1])
         #    deg_angles[k] = get_angle(deg_angles[k], deg_angles[k + 1])
         # poincare[i, j] = index
         # # print(deg_angles)
         # # print(index)
         # # if(abs(index) > 2):
         #    # print("--------WOWW\n")
         # # print("     ", i,j)
         # sys.stdout.flush()

   return poincare


def region_of_interest(img, blk_sz, gray_out=125):
   # number of blocks in a dimension
   blk_no_y, blk_no_x = (int(img.shape[0]//blk_sz), int(img.shape[1]//blk_sz))
   blk_mean = np.zeros((int(img.shape[0]//blk_sz), int(img.shape[1]//blk_sz)))
   blk_std = np.zeros((int(img.shape[0]//blk_sz), int(img.shape[1]//blk_sz)))
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

def sobel_filter(img, axis):
   img = img.astype(np.float)
   
   if axis == 0:
      # x axis
      # wikipedia
      # kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
      # scipy/cv2
      kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float)
   elif axis == 1:
      # y axis
      # wikipedia
      # kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float)
      # scipy/cv2
      kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)

   convolved = signal.convolve2d(
       img, kernel, mode='same', boundary='wrap', fillvalue=0)

   return convolved
