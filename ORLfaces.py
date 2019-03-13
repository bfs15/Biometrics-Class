#!/usr/bin/python

import cv2
import os
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceDetector = cv2.CascadeClassifier(cascadePath)

def load(path='orl_faces'):
   # face images array
   images = []
   # subject number array
   labels = []

   subjects_paths = [os.path.join(path, d) for d in os.listdir(
      path) if os.path.isdir(os.path.join(path, d))]
   for s, subject_paths in enumerate(subjects_paths, start=1):

      # Get the label of the image
      label = int(os.path.split(subject_paths)[1].split(".")[0].replace("s", ""))

      subject_path = [os.path.join(subject_paths, f) for f in os.listdir(
         subject_paths) if f.endswith('.pgm') and os.path.isfile(os.path.join(subject_paths, f))]

      for image_path in subject_path:
         print('load: {0}'.format(image_path))
         # Read the image and convert to grayscale
         image_pil = Image.open(image_path).convert('L')
         # Convert the image format into numpy array
         image = np.array(image_pil, 'uint8')  # normalization
         faces = faceDetector.detectMultiScale(image)
         for (x, y, w, h) in faces:
            face = image[y: y + h, x: x + w]
            face = cv2.resize(face, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)
            images.append(face)
            labels.append(label)
   return images, labels
