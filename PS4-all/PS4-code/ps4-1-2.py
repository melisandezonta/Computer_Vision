import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from Functions import compute_gradients, Harris_corner_response, colors_scaling




os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS4-all/PS4-input/")

images = ["transA","transB","simA","simB"]


plt.figure(figsize=(8,8))

for i in range(len(images)):

    image = cv2.cvtColor(cv2.imread(images[i]+".jpg"), cv2.COLOR_BGR2GRAY)
    m, n = image.shape


    # Computing X and Y gradient

    Ix, Iy = compute_gradients(image, 3)


    # Window chosen as a smoother Gaussian that is higher at the middle and falls off gradually

    window = cv2.getGaussianKernel(5, -1) * cv2.getGaussianKernel(5, -1).T

    # Compute the Harris corner function

    R = Harris_corner_response(Ix,Iy,0.04,window)

    # Scale of the colors

    R = colors_scaling(R)

    # Diverse plots

    plt.subplot(2, 4, i+1)
    plt.title("{} original image".format(images[i]))
    plt.imshow(image, cmap="gray")
    plt.draw()

    plt.subplot(2, 4, i+5)
    plt.title("{} Harris value".format(images[i]))
    plt.imshow(R, cmap="gray")
    cv2.imwrite("ps4-1-2-{}.png".format(images[i]), R)
    plt.draw()


plt.show()




