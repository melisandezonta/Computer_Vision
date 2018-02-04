#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:22:15 2018

@author: melisandezonta
"""

import os
import cv2
from matplotlib import pyplot as plt
from Functions import *
import numpy as np

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS1-all/PS1-images/")

# Load original image
img = cv2.imread('ps1-input0.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Load noisy image
img_noisy = cv2.imread("ps1-input0-noise.png")
img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)

# Gaussian smoothing

gaussian_kernel = cv2.getGaussianKernel(10, 2.6)
image_noisy_smoothed = cv2.sepFilter2D(img_noisy, -1, gaussian_kernel, gaussian_kernel)

# Apply the Canny edge detector on the smoothed noisy image

edges_noisy_smoothed_image = cv2.Canny(image_noisy_smoothed, 60, 100)


# Calculation of the accumulator

H = Hough_Lines(edges_noisy_smoothed_image,1)

# Visualisation of the accumulator

accumulator = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR)

# Search for the maximums


threshold = 0.4
max_points = find_max(H,threshold)
number_max_points = len(max_points[1])

# Move back in the polar domain


[m,n] = edges_noisy_smoothed_image.shape
d_max = sqrt((m-1)**2+(n-1)**2)
d = range(-ceil(d_max),ceil(d_max)+1,1)
d_hough = []
for i in range(0,number_max_points):
     d_hough.append(index_to_Hough(max_points[0][i],d))
theta_hough = max_points[1]

# Draw circles around the peaks

for i in range(0,number_max_points):
    radius = int(np.ceil(np.min([360, 360]) / 20))
    cv2.circle(accumulator, (max_points[1][i], max_points[0][i]), radius, (0,255,127))


# Draw the edge lines on the image

draw_lines(img_noisy,d_hough, theta_hough)

# Diverse plots

cv2.imwrite("ps1-3-c-accu.png", H)

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(accumulator, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps1-3-c-accu-circled.png",accumulator)

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img_noisy, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps1-3-c-lines.png", img_noisy)

plt.show()
