#!/usr/bin/python

import cv2
import os
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# load(path, ["gif", "jpeg"], True)
# loads gifs and jpegs
# load(path, ["png", "jpeg"], False)
# loads everything except "png" and "jpeg"
def load(path, filters=[], include=False):
   # face images array
   images = []
   # subject number array
   labels = []

   for filename in os.listdir(path):
      ## Filters files not supposed to open

      # default only open if extension is in the filters array
      # (open .gif and .jpeg if filters == ["gif", "jpeg"])
      openImage = False

      if not include:  # default open every file except those in the filters
         # (does NOT open .gif and .jpeg if filters == ["gif", "jpeg"])
         openImage = True

      for fil in filters:
         if filename.endswith(fil):
            # if found in filter, do opposite of default behaviour
            openImage = not openImage
      
      if not openImage:
         continue

      ## Load image

      image_path = os.path.join(path, filename)
      # Load image and convert to grayscale
      print('load: {0}'.format(image_path))
      image_pil = Image.open(image_path).convert('L')
      # Convert the image format into numpy array
      image = np.array(image_pil, 'uint8')
      # Extract label from filename (subject number)
      label = int(filename.split(".")[0].replace("subject", ""))
      # Detect the face in the image
      faces = faceCascade.detectMultiScale(image)
      # If face is detected, append the face to images and the label to labels
      for (x, y, w, h) in faces:
         face = image[y: y + h, x: x + w]
         face = cv2.resize(face, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
         images.append(face)
         labels.append(label)
   return images, labels

if __name__ == "__main__":
   # Path to the Yale Dataset, which has the face images
   path = 'yalefaces'

   print('loading yalefaces database')
   images, labels = load(path)
