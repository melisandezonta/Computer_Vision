import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS5-all/PS5-input/")

image = cv2.cvtColor(cv2.imread("DataSeq1/yos_img_01.jpg"), cv2.COLOR_BGR2GRAY)

pyramid = Gaussian_Pyramid(image, 3, cut=False)

diff_rows = pyramid[0].shape[0] - image.shape[0]
diff_cols = pyramid[0].shape[1] - image.shape[1]

laplacian_levels = []
laplacian_levels.append(pyramid[-1])
plt.figure()

for i in range(3):
    expanding = expand(pyramid[3 - i], 0.4)

    laplacian_levels.append(scale_values(pyramid[3 - (i + 1)] - expanding))

    result1 = remove_zero_padding(laplacian_levels[i], 3 - i, diff_rows, diff_cols)
    result2 = remove_zero_padding(laplacian_levels[i + 1], 3 - (i + 1), diff_rows, diff_cols)

    plt.subplot(2, 2, i + 1)
    plt.title("Level {} of the Laplacian pyramid".format(3 - i))
    plt.imshow(result1, cmap="gray")
    cv2.imwrite("ps5-1-2-{}.png".format(3 - i), result1)
    plt.draw()

    plt.subplot(2, 2, i + 2)
    plt.title("Level {} of the Laplacian pyramid".format(3 - (i + 1)))
    plt.imshow(result2, cmap="gray")
    cv2.imwrite("ps5-1-2-{}.png".format(3 - (i + 1)), result2)
    plt.draw()

plt.show()
