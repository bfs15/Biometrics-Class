
import load
import enhance
import fingerprint
import compare
import stats

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
    finger_templates = []
    blk_sz = 11

    images, subject_nos, singular_pts_list_true = load.fingerprints(
        "DB/Rindex28", blk_sz)

    mse_sum = 0
    singular_pts_correct_no = 0

    index_start = 0
    step = 2
    index_end = 4*2 # exclusive
    print("(len(images[0:]))")
    print(len(images[0:]))
    print("(len(images[index_start:]))")
    print(len(images[index_start:]))
    print(range(index_start, len(images[index_start:])))
    for image_index in range(index_start, len(images[index_start:index_end]), step):
        print(">>>\t image_index ", image_index)
        print(">\t subject_nos[image_index]", subject_nos[image_index])
        image = images[image_index]
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

        #######

        # mse = compare.points_mean_squared_error(
        #     singular_pts[0], singular_pts_list_true[image_index][1][0])
        # # print("mse=", mse)

        # mse_sum += mse

        # if(singular_type == singular_pts_list_true[image_index][0]):
        #     singular_pts_correct_no += 1
        # else:
        #     print("--- WRONG")
        #     print(singular_type, "!=", singular_pts_list_true[image_index][0])

        #####

        image_spook = fingerprint.minutiae_draw(image_spook, minutiae_list)

        image_true = image_roi.copy()
        image_draw = fingerprint.draw_orientation_map(
            image_roi, orientation_blocks, blk_sz)

        image_draw = fingerprint.draw_singular_points(
            image_draw, singular_pts, poincare, blk_sz)

        image_true = fingerprint.draw_singular_points(
            image_true, singular_pts_list_true[image_index][1], poincare, blk_sz)

        # cv2.imshow("image_draw"+str(image_index), image_draw)
        # # cv2.imshow("image_true", image_true)
        # cv2.imshow("image_spook"+str(image_index), image_spook)
        # cv2.imshow("image_smoothed"+str(image_index), image_smoothed*255)
        # sys.stdout.flush()
        # # cv2.waitKey(0)

        #######

        angle = compare.singular_orientation(singular_pts, orientation_blocks, blk_sz)

        finger_templates.append(
            (subject_nos[image_index], singular_type, singular_pts, minutiae_list, angle, image_index, image_spook, image_draw))

        print(total_time, 'total')
        print("singular_type")
        print(singular_type)
        print("singular_pts")
        print(singular_pts)
        print("angle")
        print(np.degrees(angle))
        print("----------------\n")
        sys.stdout.flush()
        
    

    def drawTemplate(template, image_index):
        template = template[image_index]
        image_spook = template[-2]
        image_draw = template[-1]

        cv2.imshow("image_draw"+str(image_index), image_draw)
        # cv2.imshow("image_true", image_true)
        cv2.imshow("image_spook"+str(image_index), image_spook)
        cv2.imshow("image_smoothed"+str(image_index), image_smoothed*255)

    print("\n--------------------------\n")

    # # Singular point MSE
    # singular_pts_accuracy = singular_pts_correct_no/len(images[0:])
    # print("singular_pts_accuracy=",singular_pts_accuracy)
    # print("mean mse=", mse_sum/len(images[0:]))

    stats.distanceMatrix(finger_templates, compare.match)

    # drawTemplate(finger_templates, 0)
    # drawTemplate(finger_templates, 4)
    # drawTemplate(finger_templates, 5)

    # cv2.waitKey(0)
    
