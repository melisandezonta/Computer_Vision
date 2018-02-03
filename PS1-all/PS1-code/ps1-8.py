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
from skimage.feature import peak_local_max



os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS1-all/PS1-images/")


# Load noisy image
img_noisy = cv2.imread("ps1-input3.jpg")
img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)


# Gaussian smoothing

image_noisy_smoothed = cv2.GaussianBlur(img_noisy_gray, (11,11), 5)

# Gradient Calculation

gxx=cv2.Sobel(image_noisy_smoothed,cv2.CV_32FC1,1,0);
gyy=cv2.Sobel(image_noisy_smoothed,cv2.CV_32FC1,0,1);
theta_test=cv2.phase(gxx,gyy,angleInDegrees=True);

# Apply the Canny edge detector on the smoothed noisy image

edges_noisy_smoothed_image = cv2.Canny(image_noisy_smoothed, 30, 50)


## Circles

# Calculation of the accumulator for the circles

[rows, cols] = edges_noisy_smoothed_image.shape
r_min = 25
r_max = 35
a_min = floor(1 - r_max)
a_max = floor(rows + r_max)
b_min = floor(1 - r_max)
b_max = floor(cols + r_max)
a_len = a_max - a_min
b_len = b_max - b_min
H_circles = Hough_Circles(edges_noisy_smoothed_image, r_min, r_max,1, a_min, a_max, b_min, b_max, a_len, b_len,theta_test)

# Search for the maximums

threshold = 0.35
max_points = find_max(H_circles,threshold)
#max_points = peak_local_max(H_circles, min_distance=1, threshold_rel=threshold, exclude_border=True)
number_max_points = len(max_points[1])

# Move back in the polar domain

[rows, cols] = edges_noisy_smoothed_image.shape
r_len = r_max - r_min
r_hough_circles = []
a_hough_circles = []
b_hough_circles = []
for i in range(0,number_max_points):
     a_hough_circles.append(int(round(a_min + max_points[0][i] * (a_max - a_min) / a_len)))
     b_hough_circles.append(int(round(b_min + max_points[1][i] * (b_max - b_min) / b_len)))
     r_hough_circles.append(int(round(r_min + max_points[2][i] * (r_max - r_min) / r_len)))

circles = zip(a_hough_circles,b_hough_circles,r_hough_circles)

# Draw the circles  on the image

img_circles = img_noisy.copy()
for a,b,r in circles:
    cv2.circle(img_circles, (a, b), r, (0,0,255), 2)


## Lines

# Calculation of the accumulator

H_lines = Hough_Lines(edges_noisy_smoothed_image,1)

# Search for the maximums

threshold = 0.55
max_points = find_max(H_lines,threshold)
number_max_points = len(max_points[1])
print(number_max_points)

# Move back in the polar domain

[m,n] = edges_noisy_smoothed_image.shape
d_max = sqrt((m-1)**2+(n-1)**2)
d = range(-ceil(d_max),ceil(d_max)+1,1)
d_hough = []
for i in range(0,number_max_points):
     d_hough.append(index_to_Hough(max_points[0][i],d))
theta_hough = max_points[1]


# Select the parallels

d_hough_lines, theta_hough_lines = find_parallels(d_hough, theta_hough, 1, 50,120)

# Draw only the parallels


draw_lines(img_circles,d_hough_lines, theta_hough_lines)

# Diverse plots

plt.subplot(2,2,1)
plt.imshow(image_noisy_smoothed, cmap='gray')
plt.draw()
cv2.imwrite("ps1-8-image-noisy-smoothed.png", image_noisy_smoothed)


plt.subplot(2,2,3)
plt.imshow(edges_noisy_smoothed_image, cmap='gray')
plt.draw()
cv2.imwrite("ps1-8-edges-image-noisy.png", edges_noisy_smoothed_image)

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps1-8-circles.png", img_circles)

plt.show()
