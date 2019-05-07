
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

def draw_singular_points(image, poincare, blk_sz, tolerance=2, thicc=2):
   '''
      image - Image array.
      poincare - Poincare index matrix of each block.
      blk_sz - block size of the orientation field.
      tolerance - Angle tolerance in degrees.
      thicc - Thickness of the lines to plot.
   '''
   # add color channels to ndarray
   image_color = np.stack((image,)*3, axis=-1)
   for i in range(1, image.shape[0]//blk_sz - 1):
      for j in range(1, image.shape[1]//blk_sz - 1):
            # if adjacent to blocks not in roi, ignore
            if(image[(i)*blk_sz, (j-1)*blk_sz] == 125 or image[(i)*blk_sz, (j+1)*blk_sz] == 125 or image[(i-1)*blk_sz, (j)*blk_sz] == 125 or image[(i+1)*blk_sz, (j)*blk_sz] == 125):
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
         y,x = point
         for point in reduces:
            y += point[0]
            x += point[1]
         reduced_point = float(y)/(len(reduces) + 1), float(x)/(len(reduces) + 1)

         new_points = points[0:i]
         new_points.append(reduced_point)
         if(next_points):
            new_points = new_points + next_points[:]
         points = new_points

      i += 1

   return points

def singular_type(image, orientation_blocks, blk_sz, tolerance=2):
   poincare = np.zeros(orientation_blocks.shape)
   blk_no_y, blk_no_x = orientation_blocks.shape

   cores = []
   deltas = []
   whorls = []
   singular_type = None
   for i in range(1, blk_no_y-1):
      for j in range(1, blk_no_x-1):
         # if adjacent to blocks not in roi, ignore
         if(image[(i)*blk_sz, (j-1)*blk_sz] == 125 or image[(i)*blk_sz, (j+1)*blk_sz] == 125 or image[(i-1)*blk_sz, (j)*blk_sz] == 125 or image[(i+1)*blk_sz, (j)*blk_sz] == 125):
            continue
         
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
            cores.append((i,j))
         angle_delta = -180
         if (np.isclose(poincare[i, j], angle_delta, 0, tolerance)):
            deltas.append((i,j))
         angle_whorl = 360
         if (np.isclose(poincare[i, j], angle_whorl, 0, tolerance)):
            whorls.append((i,j))

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
