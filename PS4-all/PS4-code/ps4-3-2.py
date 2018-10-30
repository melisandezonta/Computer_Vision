import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from Functions import *


os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS4-all/PS4-input/")

images = ["transA","transB"]

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

matches = descriptors_matching(descriptors[0], descriptors[1])

tolerance = 3
sim, sim_set = similarity_ransac(matches, keypoints[0], keypoints[1], tolerance)
percent = len(sim_set) / len(matches) * 100

print("The best similarity with {} matches ({:.1f}%) is: \n {}".format(
    len(sim_set), percent, sim))

result1 = draw_matches(images_color[0], keypoints[0], images_color[1], keypoints[1], sim_set)

result = best_match_after_similirarity(images_color[0], images_color[1], sim)

plt.title("{} and {} pair with {} marked matches ({:.1f}%) for affine transformation".format(
    images[0], images[1], len(sim_set), percent))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
cv2.imwrite("ps4-3-2-{}-{}.png".format(images[0], images[1]), result1)
plt.draw()

plt.figure()
plt.title("{} and {} pair superimposed after symmetry".format(
    images[0], images[1]))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
cv2.imwrite("ps4-3-2-{}-{}-result.png".format(images[0], images[1]), result)
plt.draw()

plt.show()




