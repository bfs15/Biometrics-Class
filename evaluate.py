
from sklearn.model_selection import KFold
import cv2
import os
import numpy as np
from PIL import Image

import yalefaces
import ORLfaces
import meanface

if __name__ == "__main__":
   # Path to the Yale Dataset
   path = 'yalefaces'
   tags = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

   # Use a cross-validation scheme of leave-one-"expression or lighting" for the Yace Face Database. What expression/lighting is the worst in terms of accuracy?
   # different lightning, different expressions are well recognized with exception os surprised

   # for tag in tags:
   #    images, labels = yalefaces.load(path, [tag], False)
   #    recognizer = meanface.EigenFaceRecognizer()
   #    recognizer.train(images, labels)
   #    print('loading database tag', tag)
   #    images, labels = yalefaces.load(path, [tag], True)
   #    print(labels)
   #    correct = 0
   #    for image, label in zip(images, labels):
   #       predicted_label = recognizer.predict_single(image)
   #       if(predicted_label == label):
   #          correct += 1
   #       # else:
   #       #    print(predicted_label, ' != ', label)
      
   #    print(correct, 'correct; accuracy: ', (float(correct) / len(images))*100, '%')

   # Use a ten-fold cross-validation scheme and report the mean an stand deviation accuracies for the ORL database. Is there a statisical significance difference between the reported values?

   from sklearn.model_selection import train_test_split
   from sklearn.model_selection import KFold, cross_val_score

   # images, labels = yalefaces.load(path, [], False)
   images, labels = ORLfaces.load()
   recognizer = meanface.EigenFaceRecognizer()
   k_fold = KFold(n_splits=10)
   print(cross_val_score(recognizer, images, labels, cv=k_fold, n_jobs=-1))
