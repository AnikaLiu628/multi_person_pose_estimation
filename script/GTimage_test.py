'''this is GT image test, confirm the kps locs are loc in right place.'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy import unravel_index
import cv2
import random

def main(_):
    keypoints = [
                332, 94, 1, 
                337, 76, 2, 
                325, 75, 1, 
                386, 78, 2,
                0, 0, 0,
                442, 154, 2,
                327, 145, 2,
                437, 273, 2, 
                291, 251, 2,
                350, 295, 2,
                227, 293, 2, 
                411, 335, 2, 
                328, 325, 2, 
                378, 471, 1, 
                216, 439, 2, 
                0, 0, 0,
                0, 0, 0]
    # store [x, y, v ...] into kx, ky, kv
    kx, ky, kv = [], [], []
    for i in range(len(keypoints)):
        if i % 3 == 0:
            kx.append(keypoints[i])
        elif i % 3 == 1:
            ky.append(keypoints[i])
        elif i % 3 == 2:
            kv.append(keypoints[i])

    #clean up kx, ky with 0
    c_kx = kx
    kx = []
    for i in range(len(c_kx)):
        if c_kx[i] == 0:
            continue
        else:
            kx.append(c_kx[i])
    c_ky = ky
    ky = []
    for i in range(len(c_ky)):
        if c_ky[i] == 0:
            continue
        else:
            ky.append(c_ky[i])

    image = cv2.imread('../data/COCO_train2014_000000113521.jpg')
    
    for i in range(len(kx)):
        cv2.circle(image, (kx[i], ky[i]), 5, (0,255,0), -1)
    print(ky)
    cv2.imshow('img',image)
    cv2.waitKey()


if __name__ == '__main__':
    tf.app.run()