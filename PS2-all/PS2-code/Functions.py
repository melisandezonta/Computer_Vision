import numpy as np
import cv2
from math import *
from time import time

def disparity_ssd(img_source, img_target, windows_size, max_d, lefttoright):
    run_start = time()
    img_source = np.matrix(img_source, dtype = "uint32")
    img_target = np.matrix(img_target, dtype = "uint32")

    rows, cols = img_source.shape

    D = np.zeros([rows, cols], dtype="uint8")

    for r in range(int(floor(windows_size / 2.0)), int(rows - floor(windows_size / 2.0))):

        for c in range(int(floor(windows_size / 2.0)), int(cols - floor(windows_size / 2.0))):

            min_r = int(r-floor(windows_size/2.0))
            max_r = int(r+ceil(windows_size/2.0))
            min_c = int(c-floor(windows_size/2.0))
            max_c = int(c+ceil(windows_size/2.0))

            source_windows = img_source[min_r:max_r, min_c:max_c]
            square_diff = []
            index_target = []

            if (lefttoright == 1):
                direction = 1
                max_displacement = int(cols - ceil(windows_size/2.0) - c)
            else:
                direction = -1
                max_displacement = int(c - floor(windows_size/2.0))


            for d in range(int(np.clip(max_d, 0, max_displacement+1))):
                target_windows = img_target[min_r:max_r, (min_c + direction * d):(max_c + direction * d)]
                square_diff.append(np.sum(np.power(source_windows - target_windows, 2)))
                index_target.append(c + direction * d)

            index_square_diff_min = index_target[square_diff.index(min(square_diff))]

            D[r, c] = direction * index_square_diff_min - direction * c
    run_end = time()
    print("The image run takes {:.3f} mins".format((run_end - run_start) / 60))
    return D


def disparity_correlation(img_source, img_target, windows_size, lefttoright):

    run_start = time()
    img_source = np.matrix(img_source, dtype = "uint32")
    img_target = np.matrix(img_target, dtype = "uint32")

    rows, cols = img_source.shape

    D = np.zeros([rows, cols], dtype="uint8")

    for r in range(int(floor(windows_size / 2.0)), int(rows - floor(windows_size / 2.0))):

        for c in range(int(floor(windows_size / 2.0)), int(cols - floor(windows_size / 2.0))):

            min_r = int(r-floor(windows_size/2.0))
            max_r = int(r+ceil(windows_size/2.0))
            min_c = int(c-floor(windows_size/2.0))
            max_c = int(c+ceil(windows_size/2.0))

            source_windows = img_source[min_r:max_r, min_c:max_c]

            if lefttoright:
                direction = 1
                target_windows = img_target[min_r:max_r,:]
            else:
                direction = -1
                target_windows = img_target[min_r:max_r,:]

            source_windows = np.asarray(source_windows,dtype ="float32")
            target_windows = np.asarray(target_windows,dtype = "float32")

            matching = cv2.matchTemplate(target_windows, source_windows, cv2.TM_CCOEFF_NORMED)
            max_index = cv2.minMaxLoc(matching)[3]

            D[r, c] = direction * max_index[0] - direction * c
    run_end = time()
    print("The image run takes {:.3f} mins".format((run_end - run_start) / 60))
    return D
