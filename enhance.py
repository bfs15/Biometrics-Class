
from PIL import Image
import numpy as np
from scipy import ndimage, misc, signal
import scipy
import matplotlib.pyplot as plt
import cv2


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
   hs = filter_size/2
   cv2.erode
   for i in range(hs, img.shape[0]-hs):
      for j in range(hs, img.shape[1]-hs):
         pixels = sorted(img[i-hs: i+hs+1, j-hs: j+hs+1].flatten())
         # dislocated median
         median[i, j] = pixels[8 + filter_size*filter_size / 2]
   return median
   # return cv2.medianBlur(img, filter_size)

def plot_point(point, angle, length, ax):
   '''
      point - Tuple (x, y) coordinates of the pixel
      angle - orientation angle at (x,y) pixel.
      length - Length of the line you want to plot.

      Will plot the line on a 10 x 10 plot.
   '''
   # unpack the first point
   x, y = point
   startx = x - np.cos(angle)*length/2
   starty = y - np.sin(angle)*length/2
   # find the end point
   endx = x + np.cos(angle)*length/2
   endy = y + np.sin(angle)*length/2
   ax.plot([startx, endx], [starty, endy], color='blue')


def draw_orientation_map(img, angles, block_size):
   row, col = img.shape
   print(img.shape)
   x_center = y_center = block_size/2  # y for rowq and x for columns
   r, c = angles.shape
   fig = plt.figure()
   ax = plt.subplot(111)
   ax.set_ylim([0, row])   # set the bounds to be 10, 10
   ax.set_xlim([0, col])
   plt.imshow(img, zorder=0, extent=[0, col, 0, row], cmap='gray')
   for i in range(0, r):
      for j in range(0, c):
         plot_point((j*block_size + y_center, i*block_size + x_center),
                     angles[i][j],
                     block_size, ax)
   plt.show()


def gradient(img, blk_sz=16):
   scipy.misc.imsave('finger.jpg', normalize(img))
   # dx = ndimage.sobel(img, 0)  # horizontal derivative
   # dy = ndimage.sobel(img, 1)  # vertical derivative
   # scipy.misc.imsave('sobel_dx.jpg', normalize(dx))
   # scipy.misc.imsave('sobel_dy.jpg', normalize(dy))

   dx = sobel_filter(img, 0)
   dy = sobel_filter(img, 1)
   scipy.misc.imsave('sobel_dy_con.jpg', normalize(dx))
   scipy.misc.imsave('sobel_dx_con.jpg', normalize(dy))

   dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
   dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
   print("dx.max()", dx.max())
   print("dy.max()", dy.max())
   scipy.misc.imsave('sobel_dy_cv2.jpg', normalize(dx))
   scipy.misc.imsave('sobel_dx_cv2.jpg', normalize(dy))


   img_alpha_x = dx**2 - dy**2
   img_alpha_y = 2 * dx * dy
   
   img_alpha_x_block = [[np.sum(img_alpha_x[index_y: index_y + blk_sz, index_x: index_x + blk_sz]) / blk_sz**2
                  for index_x in range(0, img.shape[0], blk_sz)]
                 for index_y in range(0, img.shape[1], blk_sz)]
   
   img_alpha_y_block = [[np.sum(img_alpha_y[index_y: index_y + blk_sz, index_x: index_x + blk_sz]) / blk_sz**2
                         for index_x in range(0, img.shape[0], blk_sz)]
                 for index_y in range(0, img.shape[1], blk_sz)]


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
