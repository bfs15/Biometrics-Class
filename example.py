
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

    images_enhanced = []
    block_sz = 11

    for image in images[0:]:
        image = enhance.contrast(image)
        image = enhance.median_filter(image, 5)

        orientation_blocks = enhance.gradient(image, block_sz)

        enhance.draw_orientation_map(image, orientation_blocks, block_sz)

        images_enhanced.append(image)

        # draw_orientation_map and cv2.imshow together bug out
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # plt.imshow(Image.fromarray(image), cmap='gray')
        # plt.show()
