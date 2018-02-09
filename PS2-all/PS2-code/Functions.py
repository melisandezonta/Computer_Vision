import numpy as np
import cv2
from math import *


def disparity_ssd(img_source,img_target,windows_size,lefttoright):

    img_source = np.matrix(img_source, dtype="uint32")
    img_target = np.matrix(img_target, dtype="uint32")

    rows, cols = img_source.shape

    D = np.zeros([rows,cols], dtype="uint8")

    for r in range(int(floor(windows_size/2.0)),int(rows-floor(windows_size/2.0))):

        for c in range(int(floor(windows_size/2.0)),int(cols-floor(windows_size/2.0))):

            min_r = int(floor(r - windows_size/2.0))
            max_r = int(ceil(r + windows_size/2.0))
            min_c = int(floor(c - windows_size/2.0))
            max_c = int(ceil(c + windows_size/2.0))


            source_windows = img_source[min_r:max_r,min_c:max_c]

            if lefttoright:
                direction = -1
            else:
                direction = 1


            square_diff = []
            index_target = []

            for d in range(int(floor(windows_size/2.0)),int(cols-floor(windows_size/2.0))):

                target_windows = img_target[min_r:max_r, (min_c + direction * d):(max_c + direction * d)]
                square_diff.append(np.sum(np.pow(source_windows - target_windows),2))
                index_target.append(c + direction*d)

    D[r,c] =






