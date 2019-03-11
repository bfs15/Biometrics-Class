import yalefaces
import cv2
import os
import numpy as np
from PIL import Image

# Path to the Yale Dataset
path = 'yalefaces'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
print('loading yalefaces database')
images, labels = yalefaces.load(path, ["sad"], False)

mean = np.zeros(images[0].shape)
images = images

for face in images:   
   mean += face.astype(float)/len(images)

print(mean)
mean = np.array(mean, 'uint8')
cv2.imshow("meanface >:( ", mean)
cv2.waitKey(10000)
