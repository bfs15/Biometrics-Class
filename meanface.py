import yalefaces
import cv2
import os
import numpy as np
from PIL import Image

## For face recognition we will the the LBPH Face Recognizer
recognizer1 = cv2.createLBPHFaceRecognizer()
recognizer2 = cv2.createEigenFaceRecognizer()
recognizer3 = cv2.createFisherFaceRecognizer()

# Path to the Yale Dataset
path = 'yalefaces'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
print('loading yalefaces database')
images, labels = yalefaces.load(path, ["sad"])

# Perform the tranining
print('training LBPHistogram FaceRecognizer')
recognizer1.train(images, np.array(labels))
print('training Eigen FaceRecognizer')
recognizer2.train(images, np.array(labels))
print('training Fisher FaceRecognizer')
recognizer3.train(images, np.array(labels))

images, labels = yalefaces.load(path, ["centerlight", "happy", "glasses",
                                       "leftlight", "noglasses", "normal", "rightlight", "sleepy", "surprised", "wink"])

for face, label in zip(images, labels):
    cv2.imshow("Recognizing Face", face)
    cv2.waitKey(100)
    label_predicted1, conf1 = recognizer1.predict(face)
    label_predicted2, conf2 = recognizer2.predict(face)
    label_predicted3, conf3 = recognizer3.predict(face)
    
    if label == label_predicted1:
        print("{} is Correctly Recognized with confidence {}".format(label, conf1))
    else:
        print("{} is Incorrect Recognized as {}".format(label, label_predicted1))
    if label == label_predicted2:
        print("{} is Correctly Recognized with confidence {}".format(label, conf2))
    else:
       print("{} is Incorrect Recognized as {}".format(
           label, label_predicted2))
    if label == label_predicted3:
       print("{} is Correctly Recognized with confidence {}".format(label, conf3))
    else:
       print("{} is Incorrect Recognized as {}".format(
           label, label_predicted3))
