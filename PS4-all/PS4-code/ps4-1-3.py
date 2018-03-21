import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from Functions import compute_gradients, Harris_corner_response, Harris_thresholding,colors_scaling, non_maxima_suppression, corners_markers




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

    print("Found {} corners in {} !".format(len(corners), images[i]))

    # Circle the corners on the image\

    result_image = corners_markers(corners, image)

    # Result plot

    plt.subplot(2, 2, i + 1)
    plt.title("{} with {} marked corners".format(images[i], len(corners)))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    cv2.imwrite("ps4-1-3-{}.png".format(images[i]), result_image)
    plt.draw()

plt.show()




