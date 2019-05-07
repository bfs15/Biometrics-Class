
import load
import enhance

from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import scipy
import cv2
import sys

verbose = True

if __name__ == "__main__":
    
    images = load.fingerprints("DB/Rindex28")

    images_enhanced = []
    block_sz = 11

    for image in images[0:]:
        image = enhance.contrast(image)
        image = enhance.median_filter(image, 5)

        orientation_blocks = enhance.gradient(image, block_sz)

        orientation_blocks = enhance.smooth_gradient(
            orientation_blocks, block_sz)

        image_draw = enhance.draw_orientation_map(
            image, orientation_blocks, block_sz)
        cv2.imshow("draw_orientation_map", image_draw)

        image_roi = enhance.region_of_interest(image, block_sz)

        poincare, s_type = enhance.singular_type(image_roi, orientation_blocks, block_sz)

        image_draw = enhance.draw_singular_points(image_roi, poincare, block_sz)

        image_bin = enhance.binarize(image, block_sz)

        cv2.imshow("image_bin", image_bin)
        cv2.imshow("draw_singular_points", image_draw)
        cv2.waitKey(0)
        
        images_enhanced.append(image)

        # draw_orientation_map and cv2.imshow together bug out
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # plt.imshow(Image.fromarray(image), cmap='gray')
        # plt.show()
