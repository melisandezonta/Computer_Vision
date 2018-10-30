import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS5-all/PS5-input/")

images = ["yos_img_01", "yos_img_02", "yos_img_03"]
level = 1

for i in range(len(images) - 1):
    left = cv2.imread("DataSeq1/{}.jpg".format( images[i]))
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    left = Gaussian_Pyramid(left, level, cut=True)[level]

    right = cv2.imread("DataSeq1/{}.jpg".format(images[i + 1]))
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    right = Gaussian_Pyramid(right, level, cut=True)[level]

    U, V = Lucas_Kanade(left, right, 11, 5)

    warped = warp(left, 2 * U, 2 * V)  #  warp left image to the right

    name = "ps5-2-3-{}-{}".format(images[i], images[i + 1])
    filename = "{}-displacement.png".format(name)
    fig = plt.figure()
    plt.title(filename)
    plt.imshow(left, cmap="gray")
    X = np.arange(0, U.shape[1], 2)
    Y = np.arange(0, U.shape[0], 2)
    plt.quiver(X, Y, U[::2,::2], V[::2,::2], color='r')
    fig.savefig(filename)
    plt.draw()

    filename = "{}-diff.png".format(name)
    diff = np.abs(right - warped)
    plt.figure()
    plt.title(filename)
    plt.imshow(diff, cmap="gray")
    cv2.imwrite(filename, diff)
    plt.draw()

images = ["0", "1", "2"]
level = 3

for i in range(len(images) - 1):
    left = cv2.imread("DataSeq2/{}.png".format(images[i]))
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    left = Gaussian_Pyramid(left, level, cut=True)[level]

    right = cv2.imread("DataSeq2/{}.png".format(images[i + 1]))
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    right = Gaussian_Pyramid(right, level, cut=True)[level]

    U, V = Lucas_Kanade(left, right, 5, 5)

    warped = warp(left, 2 * U, 2 * V)  #  warp left image to the right

    name = "ps5-2-3-{}-{}".format(images[i], images[i + 1])
    filename = "{}-displacement.png".format(name)
    fig = plt.figure()
    plt.title(filename)
    plt.imshow(left, cmap="gray")
    X = np.arange(0, U.shape[1], 2)
    Y = np.arange(0, U.shape[0], 2)
    plt.quiver(X, Y, U[::2,::2], V[::2,::2], color='r')
    fig.savefig(filename)
    plt.draw()

    filename = "{}-diff.png".format(name)
    diff = np.abs(right - warped)
    plt.figure()
    plt.title(filename)
    plt.imshow(diff, cmap="gray")
    cv2.imwrite(filename, diff)
    plt.draw()


plt.show()
