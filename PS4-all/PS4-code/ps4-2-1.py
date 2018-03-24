import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from Functions import *




os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS4-all/PS4-input/")

images = ["transA","transB","simA","simB"]


plt.figure(figsize=(8,8))

for i in range(len(images)):

    image = cv2.cvtColor(cv2.imread(images[i]+".jpg"), cv2.COLOR_BGR2GRAY)
    m, n = image.shape


    # Computing X and Y gradient

    I = np.zeros([m, 2 * n]) # gradients in both directions will be concatenated
    Ix, Iy = compute_gradients(image, 3)


    # Window chosen as a smoother Gaussian that is higher at the middle and falls off gradually

    window = cv2.getGaussianKernel(5, -1) * cv2.getGaussianKernel(5, -1).T

    # Compute the Harris corner function

    R = Harris_corner_response(Ix,Iy,0.04,window)

    # Scale of the colors

    R = colors_scaling(R)

    # Thresholding of the Harris corner function

    Rt = Harris_thresholding(R,50)

    # Non maxima suppression on the Harris corner function

    corners = non_maxima_suppression(Rt, 11)

    # Create the angle image
    angles = compute_angle(Ix, Iy)

    result = directions(image, corners, angles , 15)

    plt.subplot(2, 2, i + 1)
    plt.title("{} with {} markers directions".format(images[i], len(corners)))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    cv2.imwrite("ps4-2-1-{}.png".format(images[i]), result)

plt.draw()

plt.show()




