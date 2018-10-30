import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS5-all/PS5-input/")

images = ["Shift0", "ShiftR2", "ShiftR5U5"]

for i in range(1, len(images)):
    left = cv2.imread("TestSeq/{}.png".format(images[0]))
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    right = cv2.imread("TestSeq/{}.png".format(images[i]))
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    U, V = Lucas_Kanade(left, right, 21, 5)

    name = "ps5-2-1-{}-{}".format(images[0], images[i])
    filename = "{}-displacement.png".format(name)
    fig = plt.figure()
    plt.title(filename)
    plt.imshow(left, cmap="gray")
    X = np.arange(0, U.shape[1], 5)
    Y = np.arange(0, U.shape[0], 5)
    plt.quiver(X, Y, U[::5,::5], V[::5,::5], color='r')
    fig.savefig(filename)
    plt.draw()

plt.show()
