import numpy as np
import cv2
from math import *


def disparity_ssd(img_source, img_target, windows_size, lefttoright):
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
            #print(source_windows)
            square_diff = []
            index_target = []

            if lefttoright:
                direction = 1
                max_displacement = int(cols - ceil(windows_size/2.0) - c)

            else:
                direction = -1
                max_displacement = int(c - floor(windows_size/2.0))


            for d in range(0,max_displacement + 1,ceil((max_displacement + 1)/10.0)):
                #print("d is",d)
                target_windows = img_target[min_r:max_r, (min_c + direction * d):(max_c + direction * d)]
                #print(source_windows)
                #print(target_windows)
                square_diff.append(np.sum(np.power(source_windows - target_windows, 2)))
                #print("the first array is" , square_diff)
                index_target.append(c + direction * d)
                #print("the second array" , index_target)

            index_square_diff_min = index_target[square_diff.index(min(square_diff))]

            D[r, c] = direction * index_square_diff_min - direction * c

    return D
