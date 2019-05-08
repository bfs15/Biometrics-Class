
import load
import enhance
import fingerprint

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
        y = np.ma.array([1, 2, 3], mask=[0, 1, 0])
        print( y.sum())
        image = enhance.contrast(image)
        image = enhance.median_filter(image, 5)
        image_bin = enhance.binarize(image, block_sz)
        image_roi, roi_blks = enhance.region_of_interest(image, block_sz)

        orientation_blocks = fingerprint.gradient(image, block_sz)

        orientation_blocks = fingerprint.smooth_gradient(
            orientation_blocks, block_sz)

        image_draw = fingerprint.draw_orientation_map(
            image_roi, orientation_blocks, block_sz)

        poincare, s_type = fingerprint.singular_type(
            image_roi, orientation_blocks, roi_blks, block_sz)

        image_draw = fingerprint.draw_singular_points(
            image_draw, poincare, roi_blks, block_sz)

        cv2.imshow("image_draw", image_draw)
        cv2.imshow("image_bin", image_bin)
        cv2.waitKey(0)
        
        images_enhanced.append(image)

        # draw_orientation_map and cv2.imshow together bug out
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # plt.imshow(Image.fromarray(image), cmap='gray')
        # plt.show()
