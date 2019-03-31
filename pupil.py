#!/usr/bin/python

# Import the required modules
import cv2
import sys
import os
# import cv2.cv as cv
import math as mt
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy.linalg as npla
import scipy.misc as spm
import string

from utils import bilinear_interpolation

DBEyePath = "CASIA-Iris-Lamp-100"
maskFoldername = "CASIA-IrisV4-Lamp-100-mask"

IoUList = []
# timeout of imageshow when an image is not processed well
# e.g. low IoU, pupil not detected, wrong detection
waitKeyTimeoutWrong = 1

def portrait(img):
   left = 0
   bottom = img.shape[0] - 1
   right = int(img.shape[1] * 0.25)
   top = 0

   cv2.rectangle(img, (left, bottom), (right, top),
                 color=(255, 255, 255), thickness=cv2.FILLED)

   left = int(img.shape[1] * 0.75)
   bottom = img.shape[0] - 1
   right = int(img.shape[1])
   top = 0

   cv2.rectangle(img, (left, bottom), (right, top),
                 color=(255, 255, 255), thickness=cv2.FILLED)

   left = 0
   bottom = int(img.shape[0] * 0.2)
   right = int(img.shape[1])
   top = 0

   cv2.rectangle(img, (left, bottom), (right, top),
                 color=(255, 255, 255), thickness=cv2.FILLED)

   left = 0
   bottom = img.shape[0]
   right = int(img.shape[1])
   top = int(img.shape[0] * 0.8)

   cv2.rectangle(img, (left, bottom), (right, top),
                 color=(255, 255, 255), thickness=cv2.FILLED)

def contourCenter(contour):
   m = cv2.moments(contour)
   return (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))

## pupil detection
def pupil(imgEye):
   waitKeyTime = 150
   se5R = cv2.getStructuringElement(
                  cv2.MORPH_ELLIPSE, (5, 5))  # se 5x5 - Rhombus-shaped

   # opening - darkening
   ref = cv2.morphologyEx(imgEye, cv2.MORPH_OPEN, se5R, iterations=6)
#		thres_otsu,img_otsu = cv2.threshold(ref,0,255,cv2.THRESH_OTSU)
   binaryEye = np.array(np.where(ref > 30, 255, 0),
                    'uint8')  # a half of otsu value
  
   portrait(binaryEye)

   # closing
   binaryEye = cv2.morphologyEx(binaryEye, cv2.MORPH_CLOSE, se5R, iterations=7)

   # Copy to overlay edges and circles for  user visualization
   ovelayImg = imgEye.copy()

### Contours
   # inspired https://stackoverflow.com/questions/21612258/filled-circle-detection-using-cv2-in-python
   # im2, contours, hierarchy = cv2.findContours(binaryEye,
   #    cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

   # cv2.drawContours(binaryEye, contours, -1, (150, 150, 150), 2)

   # centers = []
   # radii = []
   # circles = []
   # for contour in contours:
   #    area = cv2.contourArea(contour)

   #    # there is one contour that contains all others, filter it out
   #    print('area',area)
   #    if area > 25000 or area < 5:
   #       continue

   #    br = cv2.boundingRect(contour)
   #    print('br', br, 'br[2]*1.05 > br[3]', br[2]*1.05,
   #          'br[3]*1.05 > br[2]', br[3]*1.05)
   #    sys.stdout.flush()
   #    if not (br[2]*1.50 > br[3] and br[3]*1.50 > br[2]):
   #       print('not circle enough\n')
   #       sys.stdout.flush()
   #       cv2.circle(binaryEye, contourCenter(contour), 5, (200, 200, 200), -1)
   #       cv2.imshow("Binary", binaryEye)
   #       cv2.waitKey(waitKeyTimeoutWrong)
   #    radius = int(br[2]/2)
   #    radii.append(radius)

      
   #    center = contourCenter(contour)
   #    centers.append(center)
   #    circles.append([center[0], center[1], radius])

   # # edges = np.full(binaryEye.shape, 0)
   # edges = np.zeros(binaryEye.shape, np.uint8)
   # for center, radius in zip(centers, radii):
   #    cv2.circle(ovelayImg, center, radius, (255, 255, 255), 2)
### Contours

### Canny
   edges = cv2.Canny(binaryEye, 100, 120)

   # Hough circles transform
   # param1 = 50; param2 = 30
   dp = 3.1
   minDist = 100
   param1 = 50 # 100 default
   param2 = 50  # 100 default
   # https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
   # circles = cv2.HoughCircles(
   #     edges, cv2.HOUGH_GRADIENT, dp, 20, param1=param1, param2=param2, minRadius=10, maxRadius=70)

   circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist,
                              param1=param1, param2=param2, minRadius=10, maxRadius=100)
   # make more lenient until circles are found or limit is reached
   tt = 0
   while((circles is None or ((len(circles) == 0) and (circles[0][2] == 0)) )and tt < 20):
      edges = cv2.dilate(edges, se5R, iterations=1)
      circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=10, maxRadius=100)
      param2 -= 2
      tt += 1

   if circles is not None:
      circles = circles[0]
### Canny

# display image for user
   ovelayImg[np.where(edges > 200)] = 255
   cv2.imshow("ovelayImg", ovelayImg)
#
   # if above this, not a pupil
   maxRadius = 70
   if circles is None:
      print("Failed to detect pupil")
      cv2.waitKey(waitKeyTimeoutWrong)
      return None, None
   
   # convert the (x, y) coordinates and radius of the circles to integers
   circles = np.round(circles).astype("int")
   # choose the one nearest to the center of the image
   imgCenter = np.divide(imgEye.shape, 2)
# calculate distances
   # the coordinates really are switched like this, w-why :(
   centerAux = [0, 0]
   centerAux[0], centerAux[1] = imgCenter[1], imgCenter[0]
   # algebraic distance ((a0-b0)^2 + (a1-b1)^2)^0.5
   distances = np.sum((circles[:, 0:2]-centerAux)**2, 1)**(0.5)
# equivalent code from the above snippet
   # distances = []
   # for circle in circles:
   #    # the coordinates really are switched like this, w-why :(
   #    circleY, circleX = circle[1], circle[0]
   #    imgY, imgX = imgCenter[0], imgCenter[1]
   #    # algebraic distance ((a0-b0)^2 + (a1-b1)^2)^0.5
   #    dist = ((circleY-imgY)**2 + (circleX-imgX)**2)**0.5
   #    distances.append(dist)
#
   cx, cy, pupilRadius = circles[np.argmin(distances), :]

# display image for user
   rSize = 3
   ry = int(imgCenter[0])
   rx = int(imgCenter[1])
   cv2.rectangle(ovelayImg, (rx - rSize, ry - rSize),
                  (rx + rSize, ry + rSize), 150, -1)
   for circle in circles:
      rx = circle[0]
      ry = circle[1]
      cv2.rectangle(ovelayImg, (rx - rSize, ry - rSize),
                    (rx + rSize, ry + rSize), 200, -1)
#

   imgMask = np.zeros(imgEye.shape, dtype=np.uint8)
   cv2.circle(imgMask, (cx, cy), pupilRadius, 255, 1)

# display image for user
   cv2.circle(ovelayImg, (cx, cy), pupilRadius, 255, 1)
   # circle center
   rSize = 4
   cv2.rectangle(ovelayImg, (cx - rSize, cy - rSize),
                 (cx + rSize, cy + rSize), 255, -1)
   cv2.imshow("ovelayImg", ovelayImg)
#
   if (pupilRadius < 2) or pupilRadius > maxRadius:
      print("Failed to detect pupil")
      print('circle center', (cx, cy), 'pupilRadius', pupilRadius)
      print(circles)
      print('imgCenter', imgCenter)
      print('tt', tt)
      sys.stdout.flush()
      return None, None

   return (cx, cy), pupilRadius

## iris contour detection
def irisDetect(imgEye, pupilCenter, pupilRadius):
   # Equalize histogram to increase contrast
   image = cv2.equalizeHist(imgEye)
   # graph the intensity of concentric circles around pupil of increasing radius
   intensity = []
   # initial distance
   in_dist = 20

   im_h, im_w = np.shape(imgEye)
   cx = pupilCenter[0]
   cy = pupilCenter[1]
   # max radius of the concentric circles
   maxr = min(im_h - cy, im_w - cx, cx, cy)-pupilRadius-in_dist

   # display image for user
   ovelayImg = imgEye.copy()
   cv2.circle(ovelayImg, (cx, cy), pupilRadius, 255, 1)
   cv2.imshow("ovelayImg", ovelayImg)

   # Matrix with the cosine and sines of increasing angles
   # Transpose so first dimension is [cosines,sines], and second has increasing angles
   pts_iris = np.transpose(
       [(mt.cos((mt.pi/180)*angle), mt.sin((mt.pi/180)*angle)) for angle in range(360)])
   radius = 1
   radii = range(1, maxr, 1)
   if maxr > 0:
      def _extractPixels(radius):
         # Multiply the cosines by (distance) and add center coordinates
         # to calculate pixel coordinates of a circle around cy,cx with radius=radius
         yim = (cy + pts_iris[1]*(radius)).astype(int)
         xim = (cx + pts_iris[0]*(radius)).astype(int)
         # extract pixels with coordinates [yim, xim]
         return image[yim, xim]
      # this is the initial circle
      previousCirclePixels = _extractPixels(pupilRadius+in_dist)
      for radius in radii:
         # this is the current circle with radius=radius
         currentCirclePixels = _extractPixels(pupilRadius+in_dist+radius)
         # Calculate difference in intensity, previous circle - current circle
         intensity.append(np.mean(np.abs(previousCirclePixels - currentCirclePixels)))
         previousCirclePixels = currentCirclePixels

   # Find the maximum difference in intensity
   if(len(intensity) == 0):
      cv2.waitKey(0)
   maxIntensityIndex = intensity.index(max(intensity))
   # find the radius of maximum intensity delta (and add the initial 'pupilRadius + in_dist')
   irisRadius = pupilRadius + in_dist + radii[maxIntensityIndex]

   ## create mask
   # start with 2 circles, the pupil and iris outline
   irisMask = np.zeros(imgEye.shape, dtype=np.uint8)
   cv2.circle(irisMask, (cx, cy), pupilRadius, 255, 1)
   cv2.circle(irisMask, (cx, cy), irisRadius, 255, 1)
   # floodfill the area in between
   h, w = irisMask.shape[:2]
   mask = np.zeros((h+2, w+2), np.uint8)
   cv2.floodFill(irisMask, mask, (cx+pupilRadius+1, cy), 255)

   # display overlay for user
   cv2.circle(ovelayImg, (cx, cy), irisRadius, 255, 1)
   cv2.imshow("ovelayImg", ovelayImg)

   return irisMask, irisRadius


def computeIoU(irisMask, irisMaskTrue):
   intersectionImg = irisMask.copy()
   intersectionImg[np.where(irisMaskTrue != 255)] = 0
   intersection = np.sum(intersectionImg)/255

   unionImg = irisMask.copy()
   unionImg[np.where(irisMaskTrue == 255)] = 255
   union = np.sum(unionImg)/255

   IoU = intersection / union
   if(IoU < 0.6):
      print('---Bad result---')
      sys.stdout.flush()
      cv2.imshow("irisMask", irisMask)
      cv2.imshow("irisMaskTrue", irisMaskTrue)
      cv2.imshow("intersectionImg", intersectionImg)
      cv2.imshow("unionImg", unionImg)
      cv2.waitKey(waitKeyTimeoutWrong)
   return IoU


def Mask2Norm(imgEye, irisMask, wNorm=(32, 256)):
   imgIris = imgEye.copy()
   imgIris[np.where(irisMask != 255)] = 0
   # angle points to normalize iris
   pts_norm = np.transpose([(mt.cos((2*mt.pi/wNorm[1])*angle),
                             mt.sin((2*mt.pi/wNorm[1])*angle)) for angle in range(wNorm[1])])

   cy, cx = np.divide(imgIris.shape, 2)  # extract info from irisMask
   radPupil = 20
   radIris = 40

   # imgIris = np.zeros(wNorm)
   norm_rad = []
   for i in range(wNorm[0]):
      norm_rad.append((int)(i * ((float)(radIris - radPupil)/wNorm[0]))+0.5+radPupil)

   imgNorm = np.zeros((wNorm))

   for j in range(wNorm[1]):
      for i in range(wNorm[0]):
         pt = (cy+pts_norm[1][j]*(norm_rad[i]), cx+pts_norm[0][j]*(norm_rad[i]))
         ptl = (int(mt.floor(pt[0])), int(mt.floor(pt[1])))

         pt1 = (ptl[0], ptl[1], imgIris[ptl[0], ptl[1]])
         pt2 = (ptl[0]+1, ptl[1], imgIris[ptl[0]+1, ptl[1]])
         pt3 = (ptl[0], ptl[1]+1, imgIris[ptl[0], ptl[1]+1])
         pt4 = (ptl[0]+1, ptl[1]+1, imgIris[ptl[0]+1, ptl[1]+1])

         # interpolate
         imgNorm[i][j] = bilinear_interpolation(pt[0], pt[1], [pt1, pt2, pt3, pt4])

   return imgNorm


def Normalize(imgEye, pupilCenter, pupilRadius, irisRadius, wNorm=(64, 256)):
   lines = wNorm[0]
   columns = wNorm[1]
   # sequence 0 to irisRadius distances, evenly spaced with 'lines' number of items
   radii = np.linspace(0, irisRadius, lines)
   # sequence 0 to 2*pi angles, evenly spaced with 'columns' number of angles
   angles = np.linspace(0, 2.0 * np.pi, columns+1)[:-1]
   # new polar image
   polar = np.zeros((lines, columns), dtype=np.uint8)
   for line, radius in enumerate(radii):
      print(radius)
      sys.stdout.flush()
      for col, angle in enumerate(angles):
         y = np.sin(angle) * (radius+pupilRadius) + pupilCenter[1]
         x = np.cos(angle) * (radius+pupilRadius) + pupilCenter[0]
         i = int(mt.floor(y))
         j = int(mt.floor(x))
         pt00 = (i+0, j+0, imgEye[i+0][j+0])
         pt01 = (i+0, j+1, imgEye[i+0][j+1])
         pt10 = (i+1, j+0, imgEye[i+1][j+0])
         pt11 = (i+1, j+1, imgEye[i+1][j+1])
         polar[line][col]=bilinear_interpolation(y, x, [pt00, pt10, pt01, pt11])
         # polar[line][col] = imgEye[int(y)][int(x)]
   return polar

def process(eyePath):
   IoU = None
   cv2.destroyAllWindows()
   try:
      maskPath = maskFoldername + os.sep + \
         os.sep.join(eyePath.split(os.sep)[1:])
      irisMaskTrue = Image.open(maskPath)
      irisMaskTrue = np.array(irisMaskTrue, 'uint8')
      # cv2.imshow("irisMaskTrue", irisMaskTrue)
   except FileNotFoundError as error:
      print("couldn't calculate IoU")
      print(error)
      return None

   # Read the image and convert to grayscale
   imgEye = Image.open(eyePath).convert('L')
   # Convert the image format into numpy array
   imgEye = np.array(imgEye, 'uint8')
   # self.EyeImages.append(imgEye)
   # self.idEye.append(idEye)
   pupilCenter, pupilRadius = pupil(imgEye)
   if(pupilCenter is None):
      print("Failed to detect pupil")
      cv2.waitKey(waitKeyTimeoutWrong)
      return None

   irisMask, irisRadius = irisDetect(imgEye, pupilCenter, pupilRadius)
   # cv2.imshow("irisMask", irisMask)
   # cv2.waitKey(300)


   IoU = computeIoU(irisMask, irisMaskTrue)
   IoUList.append(IoU)
   print('IoU:', IoU)
   # imgMask, imgIris = self.SegIris(imgEye)
   # if pathMask:
   #    imgpathMask = os.path.join(pathMask, os.path.split(subject_paths)[
   #                               1], os.path.split(side_path)[1], os.path.split(eyePath)[1])
   #    cv2.imwrite(imgpathMask, imgMask)
   # if pathIris:
   #    imgpathIris = os.path.join(pathIris, os.path.split(subject_paths)[
   #                               1], os.path.split(side_path)[1], os.path.split(eyePath)[1])
   #    cv2.imwrite(imgpathIris, imgIris)

   # Mask2Norm(imgEye, irisMask, (32, 256))
   polar = Normalize(imgEye, pupilCenter, pupilRadius, irisRadius)
   cv2.imshow("polar", polar)
   cv2.waitKey(100)

   # imgNorm = self.Mask2Norm(imgIris, imgMask, (32, 256))
   # if pathNorm:
   #    imgpathNorm = os.path.join(pathNorm, os.path.split(subject_paths)[
   #                               1], os.path.split(side_path)[1], os.path.split(eyePath)[1])
   #    cv2.imwrite(imgpathNorm, imgNorm)

#					fig, aplt = plt.subplots(2,2)
#					aplt[0,0].imshow(imgEye,cmap='Greys_r')
#					aplt[0,1].imshow(imgMask,cmap='Greys_r')
#					aplt[1,0].imshow(imgIris,cmap='Greys_r')
#					aplt[1,1].imshow(imgNorm,cmap='Greys_r')
#					plt.pause(_waitingtime)
#					plt.close()

   # self.IrisImages.append(imgIris)
   # self.MaskImages.append(imgMask)
   # self.NormImages.append(imgNorm)

   return IoU

if __name__ == "__main__":
   pathEye = DBEyePath
   subjects_paths = [os.path.join(pathEye, d) for d in os.listdir(
                  pathEye) if os.path.isdir(os.path.join(pathEye, d))]

   for s, subject_paths in enumerate(subjects_paths, start=1):
      # Get the label of the subject
      nsb = int(os.path.split(subject_paths)[1])

      # if pathMask and not os.path.exists(os.path.join(pathMask, os.path.split(subject_paths)[1])):
      #    os.makedirs(os.path.join(pathMask, os.path.split(subject_paths)[1]))
      # if pathIris and not os.path.exists(os.path.join(pathIris, os.path.split(subject_paths)[1])):
      #    os.makedirs(os.path.join(pathIris, os.path.split(subject_paths)[1]))
      # if pathNorm and not os.path.exists(os.path.join(pathNorm, os.path.split(subject_paths)[1])):
      #    os.makedirs(os.path.join(pathNorm, os.path.split(subject_paths)[1]))

      side_paths = [os.path.join(subject_paths, d) for d in os.listdir(
                           subject_paths) if os.path.isdir(os.path.join(subject_paths, d))]
      for e, side_path in enumerate(side_paths, start=1):
         # L or R
         sideLetter = os.path.split(side_path)[-1]

         idEye = 2*nsb + (-1 if sideLetter == 'L' else 0)
         # Print current subject and folder
         print('\t\t>{0}/{1}:{2}'.format(nsb, idEye, sideLetter))
         sys.stdout.flush()

#          if pathMask and not os.path.exists(os.path.join(pathMask, os.path.split(subject_paths)[1], os.path.split(side_path)[1])):
#             os.makedirs(os.path.join(pathMask, os.path.split(
#                                           subject_paths)[1], os.path.split(side_path)[1]))
#          if pathIris and not os.path.exists(os.path.join(pathIris, os.path.split(subject_paths)[1], os.path.split(side_path)[1])):
#             os.makedirs(os.path.join(pathIris, os.path.split(
#                                           subject_paths)[1], os.path.split(side_path)[1]))
#          if pathNorm and not os.path.exists(os.path.join(pathNorm, os.path.split(subject_paths)[1], os.path.split(side_path)[1])):
#             os.makedirs(os.path.join(pathNorm, os.path.split(
#                                           subject_paths)[1], os.path.split(side_path)[1]))

         eyePaths = [os.path.join(side_path, f) for f in os.listdir(
                                 side_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(side_path, f))]
         for y, eyePath in enumerate(eyePaths, start=1):
            # Print current filename
            print('\t-{0}:{1}'.format(y, eyePath))
            sys.stdout.flush()
            process(eyePath)
   print(IoUList)

   print('mean:', np.mean(np.array(IoUList)))
   print('var:', np.var(np.array(IoUList)))
