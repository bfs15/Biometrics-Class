
from PIL import Image
import numpy as np
from scipy import ndimage, misc, signal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm
import cv2
import math
import sys

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
         length = block_size  # total length of the line
         # down, left # add on y, sub on x
         starty = int(y + np.sin(angle)*length/2)
         startx = int(x - np.cos(angle)*length/2)
         # find the end point
         # up, right # sub on y, add on x
         endy = int(y - np.sin(angle)*length/2)
         endx = int(x + np.cos(angle)*length/2)
         cv2.line(img_draw, (startx, starty), (endx, endy), (255, 0, 0), thicc)

   return img_draw


def draw_singular_points(image, poincare, roi_blks, blk_sz, tolerance=2, thicc=2):
   '''
      image - Image array.
      poincare - Poincare index matrix of each block.
      blk_sz - block size of the orientation field.
      tolerance - Angle tolerance in degrees.
      thicc - Thickness of the lines to plot.
   '''
   # add color channels to ndarray
   if(len(image.shape) == 2):
      image_color = np.stack((image,)*3, axis=-1)
   else:
      image_color = image
   for i in range(1, image.shape[0]//blk_sz - 1):
      for j in range(1, image.shape[1]//blk_sz - 1):
            # if adjacent to blocks not in roi, ignore
            if(np.any(roi_blks[(i)*blk_sz, (j-1)*blk_sz] == 1) or np.any(roi_blks[(i)*blk_sz, (j+1)*blk_sz] == 1) or np.any(roi_blks[(i-1)*blk_sz, (j)*blk_sz] == 1) or np.any(roi_blks[(i+1)*blk_sz, (j)*blk_sz] == 1)):
               continue

            angle_core = 180
            color = matplotlib.cm.hot(abs(poincare[i, j]/360))
            color = (color[0]*255, color[1]*255, color[2]*255)
            if (np.isclose(poincare[i, j], angle_core, 0, tolerance)):
               cv2.circle(image_color, (int((j+0.5)*blk_sz), int((i+0.5) * blk_sz)),
                          int(blk_sz/2), color, thicc)

            angle_delta = -180
            if (np.isclose(poincare[i, j], angle_delta, 0, tolerance)):
               cv2.rectangle(image_color, (j*blk_sz, i*blk_sz),
                             ((j+1)*blk_sz, (i+1)*blk_sz), (255, 125, 0), thicc)

            whorl = 360
            if (np.isclose(poincare[i, j], whorl, 0, tolerance)):
               cv2.circle(image_color, (int((j+0.5)*blk_sz), int((i+0.5) * blk_sz)),
                          int(blk_sz/2), (0, 200, 200), thicc)
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


def reduce_points(points):
   i = 0
   length = len(points)
   next_points = []
   while i < length:
      next_points = []
      reduces = []
      point = points[i]
      j = 1
      while (i+j) < len(points):
         point_o = points[i+j]
         # vertical down
         if((point[1] == point_o[1] and (point[0] + 1 == point_o[0]))
            # horizontal right
            or (point[0] == point_o[0] and (point[1] + 1 == point_o[1]))
            # diagonal down right
            or (point[0]+1 == point_o[0] and point[1]+1 == point_o[1])):

            reduces.append(point_o)
         else:
            next_points.append(point_o)

         j += 1

      if(reduces):
         length = length - len(reduces)
         y, x = point
         for point in reduces:
            y += point[0]
            x += point[1]
         reduced_point = float(
             y)/(len(reduces) + 1), float(x)/(len(reduces) + 1)

         new_points = points[0:i]
         new_points.append(reduced_point)
         if(next_points):
            new_points = new_points + next_points[:]
         points = new_points

      i += 1

   return points


def singular_type(image, orientation_blocks, roi_blks, blk_sz, tolerance=2):
   poincare = np.zeros(orientation_blocks.shape)
   blk_no_y, blk_no_x = orientation_blocks.shape

   cores = []
   deltas = []
   whorls = []
   singular_type = None
   for i in range(1, blk_no_y-1):
      for j in range(1, blk_no_x-1):
         # if adjacent to blocks not in roi, ignore
         if(np.any(roi_blks[(i)*blk_sz, (j-1)*blk_sz] == 1) or np.any(roi_blks[(i)*blk_sz, (j+1)*blk_sz] == 1) or np.any(roi_blks[(i-1)*blk_sz, (j)*blk_sz] == 1) or np.any(roi_blks[(i+1)*blk_sz, (j)*blk_sz] == 1)):
            continue

         index = 0
         cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]

         def get_angle(left, right):
            angle = left - right
            if abs(angle) > 180:
               angle = -1 * np.sign(angle) * (360 - abs(angle))
            return angle

         deg_angles = [np.degrees(
             orientation_blocks[i - k][j - l]) % 180 for k, l in cells]
         # len(cells)-1 == 8
         for k in range(len(cells)-1):
            if abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
                  deg_angles[k + 1] += 180
            index += get_angle(deg_angles[k], deg_angles[k + 1])

         poincare[i, j] = index

         # def get_diff(x, y):
         #    if(y > x):
         #       x, y = y, x
         #    return x-y

         # if(abs(index) > 170 and abs(index) < 190):
         #    cells = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
         #    deg_angles = [np.degrees(orientation_blocks[i + k][j + l]) % 180 for k, l in cells]
         #    index = 0
         #    for k in range(len(cells)-1):
         #       # if (abs(get_angle(deg_angles[k], deg_angles[k + 1])) > 90):
         #       #    deg_angles[k + 1] += 180
         #       index += get_diff(deg_angles[k], deg_angles[k + 1])
         #       deg_angles[k] = get_diff(deg_angles[k], deg_angles[k + 1])
         #    poincare[i, j] = index
         #    print(deg_angles)
         #    print(index)
         #    if(abs(index) > 2):
         #       print("^.^.^.^.^.^ Found something")
         #    print("     ", i, j)
         #    print("")
         #    sys.stdout.flush()

         ### type classification

         angle_core = 180
         if (np.isclose(poincare[i, j], angle_core, 0, tolerance)):
            cores.append((i, j))
         angle_delta = -180
         if (np.isclose(poincare[i, j], angle_delta, 0, tolerance)):
            deltas.append((i, j))
         angle_whorl = 360
         if (np.isclose(poincare[i, j], angle_whorl, 0, tolerance)):
            whorls.append((i, j))

   cores = reduce_points(cores)
   deltas = reduce_points(deltas)
   whorls = reduce_points(whorls)
   print(cores)
   print(deltas)
   print(whorls)

   def points_position(cores, deltas, tolerance=2):
      if(not deltas or not cores):
         return 'middle'

      delta = deltas[0]
      core = cores[0]
      print(delta[1], core[1])
      if(np.isclose(delta[1], core[1], 0, tolerance)):
         return 'middle'

      if(delta[1] < core[1]):
         return 'left'
      else:
         return 'right'

   if(whorls or len(cores) == 2 or len(deltas) == 2):
      singular_type = ('whorl', cores, deltas, whorls)
   elif (len(deltas) <= 1 and len(cores) <= 1):
      position = points_position(cores, deltas)
      if(position == 'middle'):
         singular_type = ('arch', cores, deltas, whorls)

      if(position == 'right'):
         singular_type = ('left_loop', cores, deltas, whorls)

      if(position == 'left'):
         singular_type = ('right_loop', cores, deltas, whorls)
   else:
         singular_type = ('other', cores, deltas, whorls)

   print(singular_type)
   return poincare, singular_type

