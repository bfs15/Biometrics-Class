
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
import time

verbose = True

if __name__ == "__main__":
    
    images, subject_nos, singular_pts  = load.fingerprints("DB/Rindex28")

    images_templated = []
    blk_sz = 11

    for image in images[0:]:
        total_time = 0

        start_time = time.time()

        image = enhance.contrast(image)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'enhance.contrast')
        total_time += elapsed_time

        start_time = time.time()

        image = enhance.median_filter(image, 5)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'enhance.median_filter')
        total_time += elapsed_time

        start_time = time.time()

        image_bin = enhance.binarize(image)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'enhance.binarize')
        total_time += elapsed_time

        start_time = time.time()

        # cv2.imshow("image_bin", image_bin)
        image_spook = np.where(image_bin < 255, 1, 0).astype('uint8')

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'np.where(image_bin < 255, 1, 0)')
        total_time += elapsed_time

        start_time = time.time()

        image_smoothed = enhance.smooth_bin(image_spook, blk_sz)
        # cv2.imshow("image_smoothed", image_smoothed*255)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'enhance.smooth_bin')
        total_time += elapsed_time

        start_time = time.time()

        image_spook = skeletonize(image_smoothed).astype('uint8')

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'skeletonize')
        total_time += elapsed_time

        start_time = time.time()

        image_roi, roi_blks = enhance.region_of_interest(image, blk_sz)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'enhance.region_of_interest')
        total_time += elapsed_time

        start_time = time.time()

        orientation_blocks = fingerprint.gradient(image, blk_sz)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'fingerprint.gradient')
        total_time += elapsed_time

        start_time = time.time()

        orientation_blocks = fingerprint.smooth_gradient(
            orientation_blocks, blk_sz)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'fingerprint.smooth_gradient')
        total_time += elapsed_time

        start_time = time.time()

        poincare, singular_type, singular_pts = fingerprint.singular_pts(
            image_roi, orientation_blocks, roi_blks, blk_sz)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'fingerprint.singular_type')
        total_time += elapsed_time


        start_time = time.time()

        minutiae_list = fingerprint.minutiae(image_spook, roi_blks, blk_sz)

        elapsed_time = time.time() - start_time
        print(elapsed_time, 'fingerprint.minutiae')
        total_time += elapsed_time

        image_spook = fingerprint.minutiae_draw(image_spook, minutiae_list)


        image_draw = fingerprint.draw_orientation_map(
            image_roi, orientation_blocks, blk_sz)

        image_draw = fingerprint.draw_singular_points(
            image_draw, singular_pts, poincare, blk_sz)

        cv2.imshow("image_draw", image_draw)
        cv2.imshow("image_spook", image_spook)
        cv2.waitKey(0)
        print(total_time, 'total')
        print(singular_type)
        print(singular_pts)
        print("")
        sys.stdout.flush()
        
        images_templated.append(
            (image_spook, singular_type, singular_pts, minutiae_list))
