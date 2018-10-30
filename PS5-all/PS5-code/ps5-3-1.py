import os
import numpy as np
from math import *
import cv2
from matplotlib import pyplot as plt
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS5-all/PS5-input/")


##############################################################################
#                                 TestSeq
##############################################################################

images = ["Shift0", "ShiftR10", "ShiftR20", "ShiftR40"]
n = 4
a = 0.04
size = 5
sigma = 3

L = cv2.imread("TestSeq/{}.png".format(images[0]))
L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)

for i in range(1, len(images)):

    R = cv2.imread("TestSeq/{}.png".format(images[i]))
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    U, V, Wk, Lk, Rk = Hierarchical_LK(L, R, n, a, size, sigma)

    name = "ps5-3-1-{}-{}".format(images[0], images[i])
    filename = "{}-displacement.png".format(name)
    fig = plt.figure()
    plt.title(filename)
    plt.imshow(Lk, cmap="gray")
    X = np.arange(0, U.shape[1], 5)
    Y = np.arange(0, U.shape[0], 5)
    plt.quiver(X, Y, U[::5,::5], V[::5,::5], color='r')
    fig.savefig(filename)
    plt.draw()

    filename = "{}-diff.png".format(name)
    diff = np.abs(Rk - Wk)
    plt.figure()
    plt.title(filename)
    plt.imshow(diff, cmap="gray")
    cv2.imwrite(filename, diff)
    plt.draw()


##############################################################################
#                                 DataSeq1
##############################################################################

images = ["yos_img_01", "yos_img_02", "yos_img_03"]
n = 4
a = 0.04
size = 5
sigma = 3

for i in range(0, len(images)-1):

    L = cv2.imread("DataSeq1/{}.jpg".format(images[i]))
    L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)

    R = cv2.imread("DataSeq1/{}.jpg".format(images[i+1]))
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    U, V, Wk, Lk, Rk = Hierarchical_LK(L, R, n, a, size, sigma)

    name = "ps5-3-1-{}-{}".format(images[i], images[i+1])
    filename = "{}-displacement.png".format(name)
    fig = plt.figure()
    plt.title(filename)
    plt.imshow(Lk, cmap="gray")
    X = np.arange(0, U.shape[1], 5)
    Y = np.arange(0, U.shape[0], 5)
    plt.quiver(X, Y, U[::5,::5], V[::5,::5], color='r')
    fig.savefig(filename)
    plt.draw()

    filename = "{}-diff.png".format(name)
    diff = np.abs(Rk - Wk)
    plt.figure()
    plt.title(filename)
    plt.imshow(diff, cmap="gray")
    cv2.imwrite(filename, diff)
    plt.draw()


##############################################################################
#                                 DataSeq2
##############################################################################

images = ["0", "1", "2"]
n = 4
a = 0.04
size = 5
sigma = 3

for i in range(0, len(images)-1):

    L = cv2.imread("DataSeq2/{}.png".format(images[i]))
    L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)

    R = cv2.imread("DataSeq2/{}.png".format(images[i+1]))
    R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    U, V, Wk, Lk, Rk = Hierarchical_LK(L, R, n, a, size, sigma)

    name = "ps5-3-1-{}-{}".format(images[i], images[i+1])
    filename = "{}-displacement.png".format(name)
    fig = plt.figure()
    plt.title(filename)
    plt.imshow(Lk, cmap="gray")
    X = np.arange(0, U.shape[1], 5)
    Y = np.arange(0, U.shape[0], 5)
    plt.quiver(X, Y, U[::5,::5], V[::5,::5], color='r')
    fig.savefig(filename)
    plt.draw()

    filename = "{}-diff.png".format(name)
    diff = np.abs(Rk - Wk)
    plt.figure()
    plt.title(filename)
    plt.imshow(diff, cmap="gray")
    cv2.imwrite(filename, diff)
    plt.draw()

plt.show()
