import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS5-all/PS5-input/")

image = cv2.cvtColor(cv2.imread("DataSeq1/yos_img_01.jpg"), cv2.COLOR_BGR2GRAY)

levels = Gaussian_Pyramid(image, 3, cut=True)

for i in range(len(levels)):
    plt.subplot(2, 2, i + 1)
    plt.title("Level {} of the Gaussian pyramid".format(i))
    plt.imshow(levels[i], cmap="gray")
    cv2.imwrite("ps5-1-1-{}.png".format(i), levels[i])
    plt.draw()

plt.show()
