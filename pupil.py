#!/usr/bin/python

# Import the required modules
import cv2
import sys
import os
# import cv2.cv as cv
import math as mt
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
plt.style.use('dark_background')


from PIL import Image
import numpy as np
import numpy.linalg as npla
import scipy.misc as spm
import string
import time
# Wavelet
import pywt
# LBP
# import skimage
from skimage import feature

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import bilinear_interpolation

# DBEyePath = "CASIA-Iris-Lamp-100"
# maskFoldername = "CASIA-IrisV4-Lamp-100-mask"
DBEyePath = "CASIA-IrisV4-Interval"
maskFoldername = "CASIA-IrisV4-Interval-mask"

IoUList = []
data = []
labels = []
dataLBP = []
labelsLBP = []
# timeout of imageshow when an image is not processed well
# e.g. low IoU, pupil not detected, wrong detection
waitKeyTimeoutWrong = 1
# variable purely for user display
currentEyePath = ""


def portrait(img, portraitShape=[0.2, 0.8,  0.8, 0.2]):
   left = 0
   bottom = img.shape[0] - 1
   right = int(img.shape[1] * portraitShape[0])
   top = 0
   # thickness=cv2.FILLED = -1
   cv2.rectangle(img, (left, bottom), (right, top),
                 color=(255, 255, 255), thickness=-1)

   left = int(img.shape[1] * portraitShape[1])
   bottom = img.shape[0] - 1
   right = int(img.shape[1])
   top = 0

   cv2.rectangle(img, (left, bottom), (right, top),
                 color=(255, 255, 255), thickness=-1)

   left = 0
   bottom = img.shape[0]
   right = int(img.shape[1])
   top = int(img.shape[0] * portraitShape[2])

   cv2.rectangle(img, (left, bottom), (right, top),
                 color=(255, 255, 255), thickness=-1)

   left = 0
   bottom = int(img.shape[0] * portraitShape[3])
   right = int(img.shape[1])
   top = 0

   cv2.rectangle(img, (left, bottom), (right, top),
                 color=(255, 255, 255), thickness=-1)


def contourCenter(contour):
   m = cv2.moments(contour)
   return (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))


def findContoursPupil(binaryEye, overlayImg):
   # inspired https://stackoverflow.com/questions/21612258/filled-circle-detection-using-cv2-in-python
   im2, contours, hierarchy = cv2.findContours(binaryEye,
                                               cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

   cv2.drawContours(binaryEye, contours, -1, (150, 150, 150), 2)

   centers = []
   radii = []
   circles = []
   for contour in contours:
      area = cv2.contourArea(contour)
      # filter some clearly not appropriately sized areas
      if area > 25000 or area < 5:
         continue

      br = cv2.boundingRect(contour)
      # print('br', br, 'br[2]*1.05 > br[3]', br[2]*1.05,
      #       'br[3]*1.05 > br[2]', br[3]*1.05)
      # sys.stdout.flush()
      # check if square enough or too wide/tall to be pupil
      if not (br[2]*1.50 > br[3] and br[3]*1.50 > br[2]):
## display image for user
         print('not circle enough\n')
         sys.stdout.flush()
         # cv2.circle(binaryEye, contourCenter(contour), 5, (200, 200, 200), -1)
         # cv2.imshow(currentEyePath + " Contour", binaryEye)
         # cv2.waitKey(waitKeyTimeoutWrong)
##
         continue
      radius = int(br[2]/2)
      radii.append(radius)

      center = contourCenter(contour)
      centers.append(center)
      circles.append([center[0], center[1], radius])

   for center, radius in zip(centers, radii):
      cv2.circle(overlayImg, center, radius, (255, 255, 255), 2)

   return circles

def Canny(binaryEye, overlayImg):
   edges = cv2.Canny(binaryEye, 100, 120)
   # Hough circles transform
   # param1 = 50; param2 = 30
   dp = 3.1
   minDist = 100
   param1 = 50  # 100 default
   param2 = 50  # 100 default
   # https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
   # circles = cv2.HoughCircles(
   #     edges, cv2.HOUGH_GRADIENT, dp, 20, param1=param1, param2=param2, minRadius=10, maxRadius=70)
   # cv2.cv.CV_HOUGH_GRADIENT (python3) cv2.HOUGH_GRADIENT
   circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist,
                              param1=param1, param2=param2, minRadius=10, maxRadius=100)
   # make more lenient until circles are found or limit is reached
   tt = 0
   se5R = cv2.getStructuringElement(
       cv2.MORPH_ELLIPSE, (5, 5))  # se 5x5 - Rhombus-shaped
   while((circles is None or ((len(circles) == 0) and (circles[0][2] == 0)))and tt < 20):
      edges = cv2.dilate(edges, se5R, iterations=1)
      circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist,
                                 param1=param1, param2=param2, minRadius=10, maxRadius=100)
      param2 -= 2
      tt += 1
## display image for user
   # overlayImg[np.where(edges > 200)] = 255
   # cv2.imshow(currentEyePath + " overlayImg Canny", overlayImg)
##
   if circles is not None:
      circles = circles[0]

   return circles

def chooseCircle(circles, overlayImg):
   # if above this, not a pupil
   PupilRadiusMax = 70

   if ((circles is None) or (len(circles) == 0)):
      return (None, None), None

   # filter out non pupil circles, where radius is either zero or over max size
   circlesRaw = circles
   circles = np.array(circles)
   circles = circles[(circles[:, 2] > 2) & (circles[:, 2] <= PupilRadiusMax)]

   if(len(circles) == 0):
      if(len(circlesRaw) > 0):
         print("Filtered out circles", circlesRaw)
      return (None, None), None

   # convert the (x, y) coordinates and radius of the circles to integers
   circles = np.round(circles).astype("int")
   # choose the one nearest to the center of the image
   imgCenter = np.divide(overlayImg.shape, 2)
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
   # rSize = 3
   # ry = int(imgCenter[0])
   # rx = int(imgCenter[1])
   # cv2.rectangle(overlayImg, (rx - rSize, ry - rSize),
   #               (rx + rSize, ry + rSize), 150, -1)
   # for circle in circles:
   #    rx = circle[0]
   #    ry = circle[1]
   #    cv2.rectangle(overlayImg, (rx - rSize, ry - rSize),
   #                  (rx + rSize, ry + rSize), 200, -1)

   # cv2.circle(overlayImg, (cx, cy), pupilRadius, 255, 1)
   # # circle center
   # rSize = 4
   # cv2.rectangle(overlayImg, (cx - rSize, cy - rSize),
   #               (cx + rSize, cy + rSize), 255, -1)
   # cv2.imshow(currentEyePath + " overlayImg Circles", overlayImg)
#
   return (cx, cy), pupilRadius

## pupil detection
def pupil(imgEye):
   waitKeyTime = 150
   se5R = cv2.getStructuringElement(
       cv2.MORPH_ELLIPSE, (5, 5))  # se 5x5 - Rhombus-shaped

   imgEye = cv2.equalizeHist(imgEye)
   # cv2.imshow(currentEyePath + " imgEye", imgEye)

   # opening - darkening
   ref = cv2.morphologyEx(imgEye, cv2.MORPH_OPEN, se5R, iterations=6)
#		thres_otsu,img_otsu = cv2.threshold(ref,0,255,cv2.THRESH_OTSU)
   binaryEye = np.array(np.where(ref > 30, 255, 0),
                    'uint8')  # a half of otsu value
  
   portrait(binaryEye)
   # cv2.imshow(currentEyePath + " 0binaryEye", binaryEye)

   # closing
   binaryEye = cv2.morphologyEx(binaryEye, cv2.MORPH_CLOSE, se5R, iterations=7)
   # cv2.imshow(currentEyePath + " 1binaryEye", binaryEye)

   # Copy to overlay edges and circles for  user visualization
   overlayImg = imgEye.copy()

   pupilRadius = None
### Canny
   if pupilRadius is None:
      circles = Canny(binaryEye, overlayImg)
      (cx, cy), pupilRadius = chooseCircle(circles, overlayImg)
### Contours
   if pupilRadius is None:
      circles = findContoursPupil(binaryEye, overlayImg)
      (cx, cy), pupilRadius = chooseCircle(circles, overlayImg)

   return (cx, cy), pupilRadius


def extractCircleCoords(img, center, cosinesSinesForAngles, radius):
   # Multiply the cosines by 'radius' and add center coordinates
   # to calculate pixel coordinates of a circle around cy,cx with radius=radius
   yim = (center[1] + cosinesSinesForAngles[1]*(radius)).astype(int)
   xim = (center[0] + cosinesSinesForAngles[0]*(radius)).astype(int)
   # coordinates [yim, xim]
   return yim, xim

def extractCirclePixels(img, center, cosinesSinesForAngles, radius):
   coordinates = extractCircleCoords(
       img, center, cosinesSinesForAngles, radius)
   try:
      pixels = img[coordinates]
   except IndexError as error:
      print(error)
      print("img.shape", img.shape)
      print("center", center)
      print("radius", radius)
      
      print("coordinates[0]")
      print(coordinates[0])
      print(np.where(coordinates[0] >= img.shape[0]))
      print(coordinates[0][np.where(coordinates[0] >= img.shape[0])])
      print(np.where(coordinates[0] >= img.shape[0], 1, 0).sum())
      print("coordinates[1]")
      print(coordinates[1])
      print(np.where(coordinates[1] >= img.shape[1]))
      print(coordinates[1][np.where(coordinates[1] >= img.shape[1], 1, 0)])
      print(np.where(coordinates[1] >= img.shape[1], 1, 0).sum())
      raise
   return pixels

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

## display image for user
   # overlayImg = imgEye.copy()
   # cv2.circle(overlayImg, (cx, cy), pupilRadius, 255, 1)
   # cv2.imshow(currentEyePath + " overlayImg", overlayImg)
##
   # Matrix with the cosine and sines of increasing angles
   # Transpose so first dimension is [cosines,sines], and second has increasing angles
   cosinesSinesForAngles = np.transpose(
       [(mt.cos((mt.pi/180)*angle), mt.sin((mt.pi/180)*angle)) for angle in range(360)])
   radius = 1
   radii = range(1, maxr, 1)
   if maxr > 0:
      # this is the initial circle
      previousCirclePixels = extractCirclePixels(
          image, pupilCenter, cosinesSinesForAngles, pupilRadius+in_dist)
      for radius in radii:
         # this is the current circle with radius=radius
         currentCirclePixels = extractCirclePixels(
             image, pupilCenter, cosinesSinesForAngles, pupilRadius+in_dist+radius)
         # Calculate difference in intensity, previous circle - current circle
         intensity.append(np.mean(np.abs(previousCirclePixels - currentCirclePixels)))
         previousCirclePixels = currentCirclePixels

   # Find the maximum difference in intensity
   if(len(intensity) == 0):
      # cv2.waitKey(0)
      return None, None
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

## display overlay for user
   # cv2.circle(overlayImg, (cx, cy), irisRadius, 255, 1)
   # cv2.imshow(currentEyePath + " overlayImg", overlayImg)
##
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
      # cv2.imshow(currentEyePath + " irisMask", irisMask)
      # cv2.imshow(currentEyePath + " irisMaskTrue", irisMaskTrue)
      # cv2.imshow(currentEyePath + " intersectionImg", intersectionImg)
      # cv2.imshow(currentEyePath + " unionImg", unionImg)
      # cv2.waitKey(waitKeyTimeoutWrong)
   return IoU

## Transforms the iris pixels into a rectangle
def Mask2Norm(imgEye, pupilCenter, pupilRadius, irisRadius, wNorm=(32, 256)):
   lines = wNorm[0]
   columns = wNorm[1]
   # angle points to normalize iris
   pts_norm = np.transpose([(mt.cos((2*mt.pi/columns)*angle),
                             mt.sin((2*mt.pi/columns)*angle)) for angle in range(columns)])
    
   cx, cy = pupilCenter

   norm_rad = []
   for i in range(lines):
      norm_rad.append((int)(i * ((float)(pupilRadius - pupilRadius)/lines))+0.5+pupilRadius)

   imgNorm = np.zeros((wNorm))

   for j in range(columns):
      for i in range(lines):
         pt = (cy+pts_norm[1][j]*(norm_rad[i]), cx+pts_norm[0][j]*(norm_rad[i]))
         ptl = (int(mt.floor(pt[0])), int(mt.floor(pt[1])))

         pt1 = (ptl[0], ptl[1], imgEye[ptl[0], ptl[1]])
         pt2 = (ptl[0]+1, ptl[1], imgEye[ptl[0]+1, ptl[1]])
         pt3 = (ptl[0], ptl[1]+1, imgEye[ptl[0], ptl[1]+1])
         pt4 = (ptl[0]+1, ptl[1]+1, imgEye[ptl[0]+1, ptl[1]+1])

         # interpolate
         imgNorm[i][j] = bilinear_interpolation(pt[0], pt[1], [pt1, pt2, pt3, pt4])

   return imgNorm

## Transforms the iris pixels into a rectangle
def Normalize(imgEye, pupilCenter, pupilRadius, irisRadius, wNorm=(64, 256)):
   lines = wNorm[0]
   columns = wNorm[1]
   # sequence 0 to irisRadius distances, evenly spaced with 'lines' number of items
   radii = np.linspace(0, irisRadius-pupilRadius, lines)
   # sequence 0 to 2*pi angles, evenly spaced with 'columns' number of angles
   angles = np.linspace(0, 2.0 * np.pi, columns+1)[:-1]
   # new polar image
   polar = np.zeros((lines, columns), dtype=np.uint8)

   # Matrix with the cosine and sines of increasing angles
   # Transpose so first dimension is [0: cosines, 1: sines], and second has increasing angles
   # cosinesSinesForAngles[0] is an array of cosines  with increasing angles
   # cosinesSinesForAngles[1] is an array of sines    with increasing angles
   cosinesSinesForAngles = np.transpose(
       [(mt.cos(radAngle), mt.sin(radAngle)) for radAngle in angles])

   for line, radius in enumerate(radii):
      ## no interpolation
      # this is the current circle with radius=radius
      try:
         polar[line] = extractCirclePixels(imgEye, pupilCenter, cosinesSinesForAngles, radius+pupilRadius)
      except:
         print("line", line)
         print("pupilRadius", pupilRadius)
         print("radius", radii[line])
         print("radii", radii)
         print("pupilCenter[1]", pupilCenter[1])
         print("pupilCenter[1]+radius+pupilRadius",
               pupilCenter[1] + radius + pupilRadius)

         sys.stdout.flush()
         # cv2.waitKey(0)

      ## interpolation
      # for col, angle in enumerate(angles):
      #    y = np.sin(angle) * (radius+pupilRadius) + pupilCenter[1]
      #    x = np.cos(angle) * (radius+pupilRadius) + pupilCenter[0]
      #    ## interpolation
      #    # i = int(mt.floor(y))
      #    # j = int(mt.floor(x))
      #    # pt00 = (i+0, j+0, imgEye[i+0][j+0])
      #    # pt01 = (i+0, j+1, imgEye[i+0][j+1])
      #    # pt10 = (i+1, j+0, imgEye[i+1][j+0])
      #    # pt11 = (i+1, j+1, imgEye[i+1][j+1])
      #    # polar[line][col]=bilinear_interpolation(y, x, [pt00, pt10, pt01, pt11])
      #    ## no interpolation
      #    polar[line][col] = imgEye[int(y)][int(x)]
   return polar

def wavelet(image, levels=4):
   # Wavelet transform of image, and plot approximation and details
   decomposed = []
   for i in range(levels):
      coeffs2 = pywt.dwt2(image, 'haar')
      LL, (LH, HL, HH) = coeffs2
      image = LL
      decomposed.append([LL, LH, HL, HH])
## Display for user
   # titles = ['Approximation', ' Horizontal detail',
   #           'Vertical detail', 'Diagonal detail']
   # for i, img in enumerate(decomposed):
   #    LL, LH, HL, HH = img
   #    fig = plt.figure(figsize=(12, 3))
   #    for i, a in enumerate([LL, LH, HL, HH]):
   #       print(i)
   #       print(a)
   #       ax = fig.add_subplot(1, 4, i + 1)
   #       ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
   #       ax.set_title(titles[i], fontsize=10)
   #       ax.set_xticks([])
   #       ax.set_yticks([])
   #    fig.tight_layout()
   # plt.show()
## Display for user
   return decomposed

def LBPhistogram(image, numPoints=24, radius=8):
   # numPoints=10, radius=5
   # numPoints=24, radius=3
## compute the Local Binary Pattern representation
   lbp = feature.local_binary_pattern(image, numPoints,
      radius, method="uniform")
   # 'lbp' is of the same shape of the input image, each of the values inside lbp  ranges from [0, numPoints + 2]
## compute the normalized histogram of patterns
   (histLBP, _) = np.histogram(lbp.ravel(),
      bins=np.arange(0, numPoints + 3),
      range=(0, numPoints + 2), normed=True)
## display to user
   # x_pos = np.array(range(len(histLBP)))
   # plt.bar(x_pos, histLBP, align='center', alpha=0.5)
   # plt.xticks(x_pos, x_pos)
   # plt.ylabel('percentage of occurence')
   # plt.title('LBP histogram')
   # plt.show()
##
   return histLBP

def process(eyePath):
   IoU = None
   irisMaskTrue = None
   features = None
   cv2.destroyAllWindows()
## Open image
   try:
      maskPath = ""
      if DBEyePath == "CASIA-IrisV4-Interval":
         maskPath = maskFoldername + os.sep + eyePath.split(os.sep)[-1].split('.')[0] + ".tiff"
      else:
         maskPath = maskFoldername + os.sep + \
            os.sep.join(eyePath.split(os.sep)[1:])
      irisMaskTrue = Image.open(maskPath)
      irisMaskTrue = np.array(irisMaskTrue, 'uint8')
      # cv2.imshow(currentEyePath + " irisMaskTrue", irisMaskTrue)
      # cv2.waitKey(0)
   except EnvironmentError as error:
      print("Can't calculate IoU")
      print(error)
      # return None
   # Read the image and convert to grayscale
   imgEye = Image.open(eyePath).convert('L')
   # Convert the image format into numpy array
   imgEye = np.array(imgEye, 'uint8')
## Detect pupil
   pupilCenter, pupilRadius = pupil(imgEye)
   if(pupilRadius is None):
      print("Failed to detect pupil")
      cv2.waitKey(waitKeyTimeoutWrong)
      return None
## Detect iris
   irisMask, irisRadius = irisDetect(imgEye, pupilCenter, pupilRadius)
   if(irisRadius is None):
      print("Failed extract irisRadius")
      # cv2.waitKey(waitKeyTimeoutWrong)
      return None
   # cv2.imshow("irisMask", irisMask)
   # cv2.waitKey(300)
## Iris mask IoU evaluation
   if irisMaskTrue is not None:
      IoU = computeIoU(irisMask, irisMaskTrue)
      IoUList.append(IoU)
      print('IoU:', IoU)
## Save irisMask and irisImg
   # if pathMask:
   #    imgpathMask = os.path.join(pathMask, os.path.split(subject_paths)[
   #                               1], os.path.split(side_path)[1], os.path.split(eyePath)[1])
   #    cv2.imwrite(imgpathMask, irisMask)
   # if pathIris:
   #    imgpathIris = os.path.join(pathIris, os.path.split(subject_paths)[
   #                               1], os.path.split(side_path)[1], os.path.split(eyePath)[1])
   #    cv2.imwrite(imgpathIris, imgIris)
## Normalize image
   polar = Normalize(imgEye, pupilCenter, pupilRadius, irisRadius, (128, 256))
   # cv2.imshow("polar", polar)
   # cv2.waitKey(0)
## Save Normalization
   # imgNorm = self.Mask2Norm(imgIris, irisMask, (32, 256))
   # if pathNorm:
   #    imgpathNorm = os.path.join(pathNorm, os.path.split(subject_paths)[
   #                               1], os.path.split(side_path)[1], os.path.split(eyePath)[1])
   #    cv2.imwrite(imgpathNorm, imgNorm)
## Wavelet
   featuresWavelet = wavelet(polar, 4)
   # Use LH, HL, HH of the 4th level
   featuresWavelet = featuresWavelet[3][1:]
   featuresWavelet = np.array(featuresWavelet).flatten()
   featuresWavelet = np.where(featuresWavelet > 0, 1, 0)
## LBP
   featuresLBP = LBPhistogram(polar)

   return featuresWavelet, featuresLBP


if __name__ == "__main__":
   # start timer
   start = time.time()

   pathEye = DBEyePath
   subjects_paths = [os.path.join(pathEye, d) for d in os.listdir(
                  pathEye) if os.path.isdir(os.path.join(pathEye, d))]

   for subjectNo, subject_paths in enumerate(subjects_paths, start=1):
      # if(subjectNo > 20):
      #    continue
      # Get the label of the subject
      nsb = int(os.path.split(subject_paths)[1])
## Directory handling
      # if pathMask and not os.path.exists(os.path.join(pathMask, os.path.split(subject_paths)[1])):
      #    os.makedirs(os.path.join(pathMask, os.path.split(subject_paths)[1]))
      # if pathIris and not os.path.exists(os.path.join(pathIris, os.path.split(subject_paths)[1])):
      #    os.makedirs(os.path.join(pathIris, os.path.split(subject_paths)[1]))
      # if pathNorm and not os.path.exists(os.path.join(pathNorm, os.path.split(subject_paths)[1])):
      #    os.makedirs(os.path.join(pathNorm, os.path.split(subject_paths)[1]))
##
      side_paths = [os.path.join(subject_paths, d) for d in os.listdir(
                           subject_paths) if os.path.isdir(os.path.join(subject_paths, d))]
      for e, side_path in enumerate(side_paths, start=1):
         # L or R
         sideLetter = os.path.split(side_path)[-1]

         idEye = 2*nsb + (-1 if sideLetter == 'L' else 0)
         # Print current subject and folder
         print('\t\t>{0}/{1}:{2}'.format(nsb, idEye, sideLetter))
         sys.stdout.flush()
## Directory handling
         # if pathMask and not os.path.exists(os.path.join(pathMask, os.path.split(subject_paths)[1], os.path.split(side_path)[1])):
         #    os.makedirs(os.path.join(pathMask, os.path.split(
         #                                  subject_paths)[1], os.path.split(side_path)[1]))
         # if pathIris and not os.path.exists(os.path.join(pathIris, os.path.split(subject_paths)[1], os.path.split(side_path)[1])):
         #    os.makedirs(os.path.join(pathIris, os.path.split(
         #                                  subject_paths)[1], os.path.split(side_path)[1]))
         # if pathNorm and not os.path.exists(os.path.join(pathNorm, os.path.split(subject_paths)[1], os.path.split(side_path)[1])):
         #    os.makedirs(os.path.join(pathNorm, os.path.split(
         #                                  subject_paths)[1], os.path.split(side_path)[1]))
##
         eyePaths = [os.path.join(side_path, f) for f in os.listdir(
                                 side_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(side_path, f))]
         for y, eyePath in enumerate(eyePaths, start=1):
            # Print current filename
            print('\t-{0}:{1}'.format(y, eyePath))
            sys.stdout.flush()
            # extract features
            # variable purely for user display
            currentEyePath = os.sep.join(eyePath.split(os.sep)[1:])
            features = process(eyePath)
            featuresWavelet, featuresLBP = None, None
            if features is not None:
               featuresWavelet, featuresLBP = features

            if featuresWavelet is not None:
               # add to database
               data.append(featuresWavelet)
               labels.append(subjectNo)
            if featuresLBP is not None:
               # add to database
               dataLBP.append(featuresLBP)
               labelsLBP.append(subjectNo)

   end = time.time()
   print("{0:.2f}".format(round(end - start, 2)), " seconds elapsed")
## Tests
   def distanceCalc(x, y):
      # return (np.bitwise_xor(x, y).sum()) / len(y)
      return ((x != y).sum()) / len(y)
## Iris mask IoU evaluation
   IoUList = np.array(IoUList)
   print('IoU mean:', np.mean(IoUList))
   print('IoU var:', np.var(IoUList))
## Verification
   def DETCurve(fps, fns):
      """
      Given false positive and false negative rates, produce a DET Curve.
      The false positive rate is assumed to be increasing while the false
      negative rate is assumed to be decreasing.
      """
      axis_min = min(fps[0], fns[-1])
      fig, ax = plt.subplots()
      plt.xlabel("FAR")
      plt.ylabel("FRR")
      plt.plot(fps, fns, '-|')
      plt.yscale('log')
      plt.xscale('log')
      ax.get_xaxis().set_major_formatter(
         FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
      ax.get_yaxis().set_major_formatter(
         FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
      ticks_to_use = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1]
      ax.set_xticks(ticks_to_use)
      ax.set_yticks(ticks_to_use)
      plt.axis([0.01, 1, 0.01, 1])
      # plt.show()
		
   def verificationTest(distances, labels, threshold):
      falseTestsNo = 0
      positiveTestsNo = 0
      trueRejectionNo = 0
      truePositivesNo = 0
      FAR = 0
      FRR = 0
      for i in range(len(labels)):
         for j in range(len(labels)):
            if i == j:
               continue
            if labels[i] != labels[j]:
               falseTestsNo += 1
               if distances[i][j] > threshold:
                  trueRejectionNo += 1
               else:
                  FAR += 1
            else:
               positiveTestsNo += 1
               if distances[i][j] <= threshold:
                  truePositivesNo += 1
               else:
                  FRR += 1

      return FAR/float(falseTestsNo), FRR/float(positiveTestsNo)

   def verificationTestNP(distances, labels, threshold):
      falseTestsNo = 0
      positiveTestsNo = 0
      FAR = 0
      FRR = 0
      labels = np.array(labels)

      for i in range(len(labels)):
         # extract distances between different subjects
         indexes = np.where(labels != labels[i])
         falseTests = distances[i][indexes]
         # calculate how many are accepted when they should have been rejected
         FAR += (falseTests <= threshold).sum()
         # update total number of tests
         falseTestsNo += len(falseTests)

         # extract distances between same subject
         indexes = np.where(labels == labels[i])[0]
         # remove distance between same sample
         index = np.argwhere(indexes == i)
         indexes = np.delete(indexes, index)
         # extract distances between different subjects
         positiveTests = distances[i][indexes]
         # calculate how many are accepted when they should have been rejected
         FRR += (positiveTests > threshold).sum()
         positiveTestsNo += len(positiveTests)
         # print("indexes", indexes)
      return FAR/float(falseTestsNo), FRR/float(positiveTestsNo)
## Computing distance array
   # start timer
   start = time.time()

   print("Computing distance array")
   sys.stdout.flush()
   # initialize list
   distances = [None]*len(data)
   for i in range(len(data)):
      # initialize list
      distances[i] = [None]*len(data)
      for j in range(len(data)):
         distances[i][j] = distanceCalc(data[i], data[j])


   end = time.time()
   print("{0:.2f}".format(round(end - start, 2)), " seconds elapsed")
## Computing FAR, FRR
   # start timer
   start = time.time()

   print("Computing FAR, FRR")
   sys.stdout.flush()

   fpsList = []
   fnsList = []
   distances = np.array(distances)
   for threshold in np.linspace(0.1, 0.9, num=int(0.9/0.001)):
      # FAR, FRR = verificationTest(distances, labels, threshold)
      # fpsList.append(FAR)
      # fnsList.append(FRR)
      FAR, FRR = verificationTestNP(distances, labels, threshold)
      fpsList.append(FAR)
      fnsList.append(FRR)
   
   DETCurve(fpsList, fnsList)
   plt.savefig('DETCurve.png')

   diff = np.abs(np.array(fpsList) - np.array(fnsList))

   argEER = np.argmin(diff)
   EER = np.min(diff)

   print("argEER", argEER)
   print("EER", EER)
   
   print("fps", fpsList[argEER])
   print("fns", fnsList[argEER])

   end = time.time()
   print("{0:.2f}".format(round(end - start, 2)), " seconds elapsed")

## Identification
   # # start timer
   # start = time.time()

   # # test data with SVM
   # model = LinearSVC(C=100.0, random_state=42)
   # k_fold = None
   # max_fold = 40
   # n_splits = 40
   # for i in range(max_fold):
   #    try:
   #       n_splits = max_fold-i
   #       k_fold = StratifiedKFold(n_splits=n_splits)
   #       print("n_splits: \t", n_splits)
   #       print(cross_val_score(model, dataLBP, labels, cv=k_fold, n_jobs=-1))
   #       print(cross_val_score(model, data, labels, cv=k_fold, n_jobs=-1))
   #       break
   #    except ValueError as error:
   #       pass

   
   # # def testIdentification(X_train, X_test, y_train, y_test):
   # #    model.fit(X_train, y_train)
   # #    correct = 0
   # #    for i, data in enumerate(X_test):
   # #       prediction = model.predict(data)
   # #       if y_test[i] == prediction:
   # #          correct += 1
   # #    accuracy = correct / float(len(X_test))
   # #    return accuracy
   # # accuracyList = []
   # # k_fold = StratifiedKFold(n_splits=4)
   # # X = np.array(data)
   # # y = np.array(labels)
   # # k_fold.get_n_splits(X)
   # # for train_index, test_index in k_fold.split(X,labels):
   # #    # print("TRAIN:", train_index, "TEST:", test_index)
   # #    X_train, X_test = X[train_index], X[test_index]
   # #    y_train, y_test = y[train_index], y[test_index]
   # #    accuracy = testIdentification(X_train, X_test, y_train, y_test)
   # #    accuracyList.append(accuracy)
   # # accuracyList = np.array(accuracyList)
   # # print('Identification mean:', np.mean(accuracyList))
   # # print('Identification var:', np.var(accuracyList))

   # end = time.time()
   # print("{0:.2f}".format(round(end - start, 2)), " seconds elapsed")
   
## train_test_split
   # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
   # def test(X_train, X_test, y_train, y_test):
   #    correct_num = 0
   #    for x, y in zip(X_test, y_test):
   #       minDist = sys.maxint
   #       guess = None

   #       for feat, label in zip(X_train, y_train):
   #          hammingDist = distanceCalc(x, feat)
   #          if hammingDist < minDist:
   #             minDist = hammingDist
   #             guess = label
         
   #       if guess == y:
   #          correct_num += 1
   #          print('+correct')
   #       else:
   #          print('-incorrect')
         
   #       print('minDist', minDist)

   #    accuracy = correct_num/ len(X_test)
   #    print('accuracy: ', accuracy)
   #    return accuracy
   # accuracy = test(X_train, X_test, y_train, y_test)
