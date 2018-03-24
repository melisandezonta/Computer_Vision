import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from Functions import *




os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS4-all/PS4-input/")

images = ["transA","transB","simA","simB"]

images_color = []

keypoints = []

descriptors = []

plt.figure(figsize=(8,8))

for i in range(len(images)):

    image = cv2.cvtColor(cv2.imread(images[i]+".jpg"), cv2.COLOR_BGR2GRAY)

    images_color.append(image)

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

    keypoint, descriptor = descriptors_description(image,corners,angles)

    keypoints.append(keypoint)

    descriptors.append(descriptor)

matches_image_1 = descriptors_matching(descriptors[0], descriptors[1])
matches_image_2 = descriptors_matching(descriptors[2], descriptors[3])

result_image_1 = draw_matches(images_color[0], keypoints[0], images_color[1], keypoints[1], matches_image_1)
result_image_2 = draw_matches(images_color[2], keypoints[2], images_color[3], keypoints[3], matches_image_2)

plt.subplot(2,1,1)
plt.title("{} and {} pair with {} marked matches".format(images[0], images[1], len(matches_image_1)))
plt.imshow(cv2.cvtColor(result_image_1, cv2.COLOR_BGR2RGB))
cv2.imwrite("ps4-2-2-{}-{}.png".format(images[0], images[1]), result_image_1)
plt.subplot(2,1,2)
plt.title("{} and {} pair with {} marked matches".format(images[2], images[3], len(matches_image_2)))
plt.imshow(cv2.cvtColor(result_image_2, cv2.COLOR_BGR2RGB))
cv2.imwrite("ps4-2-2-{}-{}.png".format(images[2], images[3]), result_image_2)

plt.draw()

plt.show()




