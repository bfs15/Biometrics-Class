
import cv2
import os
import numpy as np
from PIL import Image

import yalefaces
import ORLfaces
import meanface

print('loading database')
# Path to the Yale Dataset
path = 'yalefaces'
tags = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

for tag in tags:
   images, labels = yalefaces.load(path, [tag], False)
   recognizer = meanface.EigenFaceRecognizer()
   recognizer.train(images, labels)
   print('loading database tag', tag)
   images, labels = yalefaces.load(path, [tag], True)
   print(labels)
   correct = 0
   for image, label in zip(images, labels):
      predicted_label = recognizer.predict(image)
      print(predicted_label, ' == ', label)
      if(predicted_label == label):
         correct += 1
   
   print(correct, 'correct; accuracy: ', (correct / len(images))*100, '%')
