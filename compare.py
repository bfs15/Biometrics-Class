
from PIL import Image
import numpy as np
import scipy
import cv2
import sys
import math


def singular_orientation(singular_pts, orientation_blocks, blk_sz):
   core_coord = singular_pts[0][0]
   blkY = int(core_coord[0]//blk_sz)
   blkX = int(core_coord[1]//blk_sz)
   angles = orientation_blocks[blkY-3-1: blkY+7 + 1, blkX: blkX+3+1]
   angle = angles.mean()

   # print("\n>> singular_orientation\n")
   # print(singular_pts)
   # print(core_coord)
   # print(blkY, blkX)
   # print(np.degrees(orientation_blocks[blkY-1: blkY+1+1, blkX-1: blkX+2+1]))
   # print(np.degrees(orientation_blocks[blkY-1: blkY + 1, blkX: blkX+2+1]))
   return angle


def centralize(template):
   minutiae_list, singular_pts = template[3], template[2]
   core_coord = np.array(singular_pts[0])
   minutiae_list_cent = []
   # print(">> Centralize")
   # print(core_coord)
   for min_typed_list in minutiae_list:
      if np.array(min_typed_list).size == 0:
         minutiae_list_cent.append(np.array([]))
         continue
      minutiae_list_cent.append(np.array(min_typed_list) - core_coord)
      # print("min_typed_list")
      # print(np.array(min_typed_list).shape)
      # print(np.array(min_typed_list))
      # print(np.array(min_typed_list) - core_coord)
   return minutiae_list_cent


def rotate(point, angle, origin=(0, 0)):
   """
   Rotate a point counterclockwise by a given angle around a given origin.
   Angles in radians.
   """
   ox, oy = origin
   px, py = point

   qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
   qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
   return qx, qy


def rotate_minutiae(minA, angleA, minB, angleB):
   angleDiff = angleA - angleB
   print("\n> angleDiff")
   print(angleDiff)

   if(abs(np.degrees(angleDiff)) > 30):
      print("--- Failed to rotate minutiae")
      print("angles too different")
      return minA, minB

   for i_type in range(len(minB)):
      min_typedB = minB[i_type]
      if len(min_typedB) == 0:
         continue

      minB[i_type] = np.array(list(map(lambda x: rotate(x, angleDiff), min_typedB)))

   return minA, minB


def match(tempA, tempB):
   print("")
   print("compare")
   print("subj_no")
   print(tempA[0], " == ", tempB[0])
   print("index")
   print(tempA[-3], " == ", tempB[-3])
   sys.stdout.flush()


   minA_cent = centralize(tempA)
   minB_cent = centralize(tempB)
   angleA, angleB = tempA[4], tempB[4]
   minA, minB = rotate_minutiae(minA_cent, angleA, minB_cent, angleB)

   # print("\n> singular pts")
   # # list of cores, deltas and whorls
   # print(tempA[2])
   # # list of coordinates ([0] = cores; [1] = deltas; [2] = whorls)
   # print(np.array(tempA[2][0]))

   score = 0
   mse, score = minutia_match(minA, minB)

   print("mse")
   print(mse)
   print("score")
   print(score)
   print("-----------")
   return mse, score


# images, subject_nos, singular_pts
def minutia_match(pointsTypedA, pointsTypedB, threshold=16):
   mse = 0
   points = 0
   match_score = 0
   
   for type_no in range(len(pointsTypedA)):
      print("> type_no")
      print(type_no)
      print("pointsTypedA[type_no].size")
      print(pointsTypedA[type_no].size)
      print("pointsTypedB[type_no].size")
      print(pointsTypedB[type_no].size)
      sys.stdout.flush()

      pointsA = pointsTypedA[type_no]
      pointsB = pointsTypedB[type_no]

      distances_shape = (len(pointsA), len(pointsB))
      distances = np.full(distances_shape, -1)
      for i_pred in range(len(pointsA)):
            for j_true in range(len(pointsB)):
               # distances[i_pred, j_true] = np.sqrt(
               #     (pred[i_pred][0]-true[j_true][0]) ** 2 + (pred[i_pred][1]-true[j_true][1]) ** 2)
               distances[i_pred, j_true] = np.linalg.norm(
                  np.array(pointsA[i_pred]) - np.array(pointsB[j_true]))

      # print("distances")
      # print(distances)
      print("distances sort")
      print(np.sort(distances.flatten()))
      sys.stdout.flush()

      dist_arg_sort = np.dstack(np.unravel_index(
         np.argsort(distances.ravel()), distances_shape))[0]
      mask = np.ones(len(dist_arg_sort), dtype=bool)
      for i_arg_dist in range(len(dist_arg_sort)):
         if(mask[i_arg_dist] == False):
            continue
         dist_smallest_arg = dist_arg_sort[i_arg_dist]
         dist_smallest = distances[dist_smallest_arg[0], dist_smallest_arg[1]]
         if(dist_smallest > threshold):
            print("--- Reached threshold after ", i_arg_dist, " points")
            print("--- distance of ", dist_smallest, " pixels")
            break

         mse += dist_smallest ** 2
         match_score += 1 - (dist_smallest / threshold)
         points += 1

         # mask used points
         maskIndexes = np.where(
            (dist_arg_sort[:, 0] == dist_smallest_arg[0]))
         mask[maskIndexes] = False

         maskIndexes = np.where(
            (dist_arg_sort[:, 1] == dist_smallest_arg[1]))
         mask[maskIndexes] = False

         mask[i_arg_dist] = True

      # # what do if there are different quantity of points?
      # if(len(pointsA) != len(pointsB)):
      #    pass
      
      print("")
      print("")
   mse = mse/(points*2)
   match_score = 100*match_score/(points)
   return mse, match_score


# images, subject_nos, singular_pts
def points_mean_squared_error(pred, true, threshold=33):
   mse = 0
   points = 0

   distances_shape = (len(pred), len(true))
   distances = np.full(distances_shape, -1)
   for i_pred in range(len(pred)):
         for j_true in range(len(true)):
            # distances[i_pred, j_true] = np.sqrt(
            #     (pred[i_pred][0]-true[j_true][0]) ** 2 + (pred[i_pred][1]-true[j_true][1]) ** 2)

            distances[i_pred, j_true] = np.linalg.norm(
               np.array(pred[i_pred]) - np.array(true[j_true]))

   ordered_arg_dist = np.dstack(np.unravel_index(
         np.argsort(distances.ravel()), distances_shape))[0]
   mask = np.ones(len(ordered_arg_dist), dtype=bool)
   for i_arg_dist in range(len(ordered_arg_dist)):
         if(mask[i_arg_dist] == False):
            continue
         arg_dist = ordered_arg_dist[i_arg_dist]
         # if(distances[arg_dist[0], arg_dist[1]] > threshold):
         #     pass
         # mask used points
         maskIndexes = np.where(
            (ordered_arg_dist[:, 0] == arg_dist[0]))
         mask[maskIndexes] = False

         maskIndexes = np.where(
            (ordered_arg_dist[:, 1] == arg_dist[1]))
         mask[maskIndexes] = False

         mask[i_arg_dist] = True

         mse += distances[arg_dist[0], arg_dist[1]] ** 2
         points += 1

   # if(len(pred) != len(true)):
   #     pass

   return mse/(points*2)
