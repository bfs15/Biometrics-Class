
import load
import enhance

from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import scipy
import cv2

verbose = True

if __name__ == "__main__":
    
    images = load.fingerprints("DB/Rindex28")

    if (verbose):
        print(type(images[0]))
        print('max', np.array(images[0]).max())
        print('min', np.array(images[0]).min())

    images_enhanced = []
    block_sz = 11

    for image in images[0:]:
        image = enhance.contrast(image)
        median = enhance.median_filter(image, 5)

        scipy.misc.imsave('median.jpg', median)

        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.dilate(image, kernel, iterations=1)
        scipy.misc.imsave('erosion.jpg', erosion)

        orientation_blocks = enhance.gradient(image, block_sz)
        enhance.draw_orientation_map(
            image, orientation_blocks, block_sz)

        images_enhanced.append(image)

    image = images_enhanced[0]

    if (verbose):
        print("max", image.max())
        print("min", image.min())
        print(image)
        # plt.imshow(Image.fromarray(image), cmap='gray')
        # plt.show()
