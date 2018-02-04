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


# Load noisy image
img_noisy = cv2.imread("ps1-input1.jpg")
img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)


# Gaussian smoothing

image_noisy_smoothed = cv2.GaussianBlur(img_noisy_gray, (3,3), 5)

# Gradient Calculation

gxx=cv2.Sobel(image_noisy_smoothed,cv2.CV_32FC1,1,0);
gyy=cv2.Sobel(image_noisy_smoothed,cv2.CV_32FC1,0,1);
theta_test=cv2.phase(gxx,gyy,angleInDegrees=True);



# Apply the Canny edge detector on the smoothed noisy image

edges_noisy_smoothed_image = cv2.Canny(image_noisy_smoothed, 60, 100)


# Calculation of the accumulator

[rows, cols] = edges_noisy_smoothed_image.shape
r_min = 20
r_max = 30
a_min = floor(1 - r_max)
a_max = floor(rows + r_max)
b_min = floor(1 - r_max)
b_max = floor(cols + r_max)
a_len = a_max - a_min
b_len = b_max - b_min
H = Hough_Circles(edges_noisy_smoothed_image, r_min, r_max,1, a_min, a_max, b_min, b_max, a_len, b_len,theta_test)

# Search for the maximums

threshold = 0.45
max_points = find_max(H,threshold)
number_max_points = len(max_points[1])
print('the number of max_points is :', number_max_points)

# Move back in the polar domain

[rows, cols] = edges_noisy_smoothed_image.shape
r_len = r_max -r_min
r_hough = []
a_hough = []
b_hough = []
distance = 5
for i in range(0,number_max_points):
     a_hough.append(int(round(a_min + max_points[0][i] * (a_max - a_min) / a_len)))
     b_hough.append(int(round(b_min + max_points[1][i] * (b_max - b_min) / b_len)))
     r_hough.append(int(round(r_min + max_points[2][i] * (r_max - r_min) / r_len)))
a_new_hough, b_new_hough, r_new_hough = filter_circles(a_hough,b_hough,r_hough,40.0)
circles = zip(a_new_hough,b_new_hough,r_new_hough)

# Draw the circles  on the image

img_circles = cv2.cvtColor(image_noisy_smoothed, cv2.COLOR_GRAY2BGR)
for a,b,r in circles:
    cv2.circle(img_circles, (a, b), r, (0,255,127), 2)

# Diverse plots

plt.subplot(2,2,1)
plt.imshow(image_noisy_smoothed, cmap='gray')
plt.draw()
cv2.imwrite("ps1-5-image-noisy-smoothed.png", image_noisy_smoothed)


plt.subplot(2,2,2)
plt.imshow(edges_noisy_smoothed_image, cmap='gray')
plt.draw()
cv2.imwrite("ps1-5-edges-image-noisy.png", edges_noisy_smoothed_image)

plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps1-5-circles.png", img_circles)

plt.show()
