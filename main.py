
import load
import enhance
import fingerprint

from PIL import Image
import numpy as np
from matplotlib import pylab as plt
import scipy
import cv2
import sys
from skimage.morphology import skeletonize
import matplotlib.cm

verbose = True

if __name__ == "__main__":
    
    images = load.fingerprints("DB/Rindex28")

    images_enhanced = []
    blk_sz = 11

    for image in images[0:]:
        y = np.ma.array([1, 2, 3], mask=[0, 1, 0])
        print( y.sum())
        image = enhance.contrast(image)
        image = enhance.median_filter(image, 5)
        image_bin = enhance.binarize(image)
        cv2.imshow("image_bin", image_bin)
        image_spook = np.where(image_bin < 255, 1, 0).astype('uint8')
        image_smoothed = enhance.smooth_bin(image_spook, blk_sz)
        image_spook = skeletonize(image_smoothed).astype('uint8')

        image_roi, roi_blks = enhance.region_of_interest(image, blk_sz)

        orientation_blocks = fingerprint.gradient(image, blk_sz)

        orientation_blocks = fingerprint.smooth_gradient(
            orientation_blocks, blk_sz)

        poincare, s_type = fingerprint.singular_type(
            image_roi, orientation_blocks, roi_blks, blk_sz)

        minutiae_list = fingerprint.minutiae(image_spook, roi_blks, blk_sz)

        image_spook = fingerprint.minutiae_draw(image_spook, minutiae_list)

        image_draw = fingerprint.draw_orientation_map(
            image_roi, orientation_blocks, blk_sz)

        image_draw = fingerprint.draw_singular_points(
            image_draw, s_type, poincare, blk_sz)

        print("minutiae_list")
        print(minutiae_list)
        
        cv2.imshow("image_draw", image_draw)
        cv2.imshow("image_smoothed", image_smoothed*255)
        cv2.imshow("image_spook", image_spook)
        cv2.waitKey(0)
        
        images_enhanced.append(image)

        # draw_orientation_map and cv2.imshow together bug out
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # plt.imshow(Image.fromarray(image), cmap='gray')
        # plt.show()
