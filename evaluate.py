
from sklearn.model_selection import KFold
import cv2
import os
import numpy as np
from PIL import Image
from scipy import stats

import yalefaces
import ORLfaces
import eigenface

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

## sklearn interface with cv implementation
# to use cross_val_score

class cv2EigenFaceRecognizer(BaseEstimator, ClassifierMixin):
   def __init__(self):
      super(cv2EigenFaceRecognizer, self).__init__()
      self._recognizer = cv2.face.EigenFaceRecognizer_create()

   def fit(self, X, y=None):
      self._recognizer.train(X, np.array(y))

   def predict_list(self, faces):
      y_pred = []
      for face in faces:
         y_pred.append(self._recognizer.predict(face)[0])
      return y_pred

   def score(self, X, y):
      y_pred = self.predict_list(X)
      return accuracy_score(y_pred, y)


if __name__ == "__main__":
   # Path to the Yale Dataset
   path = 'yalefaces'
   tags = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses',
           'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
   # tags = ['sad']

   # Use a cross-validation scheme of leave-one-"expression or lighting" for the Yace Face Database. What expression/lighting is the worst in terms of accuracy?
   # different lightning, different expressions are well recognized with exception os surprised

   predictions = []
   for tag in tags:
      images, labels = yalefaces.load(path, [tag], False)
      recognizer = eigenface.EigenFaceRecognizer()
      # recognizer = cv2.face.EigenFaceRecognizer_create()
      recognizer.train(images, labels)
      print('loading database tag', tag)
      images, labels = yalefaces.load(path, [tag], True)
      print(labels)
      correct = 0
      for image, label in zip(images, labels):
         predicted_label = recognizer.predict(image)
         # predicted_label = prediction[0]
         if(predicted_label == label):
            correct += 1
         # else:
         #    print(predicted_label, ' != ', label)
      mean = (float(correct) / len(images))
      print(correct, 'correct; accuracy: ', mean*100, '%')
      predictions.append(mean)

   print('mean accuracy: ', np.array(predictions).mean())
   print('stddev: ', np.array(predictions).std())

   # Use a ten-fold cross-validation scheme and report the mean an stand deviation accuracies for the ORL database. Is there a statisical significance difference between the reported values?

   from sklearn.model_selection import train_test_split
   from sklearn.model_selection import StratifiedKFold, cross_val_score

   images, labels = ORLfaces.load()
   # recognizer = eigenface.EigenFaceRecognizer()
   # recognizer.train(images, labels)

   recognizer = eigenface.EigenFaceRecognizer()
   k_fold = StratifiedKFold(n_splits=10)
   scores = cross_val_score(recognizer, images, labels, cv=k_fold, n_jobs=1)
   print(scores)
   print('mean accuracy: ', np.array(scores).mean())
   print('stddev: ', np.array(scores).std())

   scores1 = scores

   recognizer2 = cv2EigenFaceRecognizer()
   k_fold = StratifiedKFold(n_splits=10)
   scores = cross_val_score(recognizer2, images, labels, cv=k_fold, n_jobs=1)
   print(scores)
   print('mean accuracy: ', np.array(scores).mean())
   print('stddev: ', np.array(scores).std())

   # print(stats.ttest_ind(scores1, scores1, equal_var=False))

