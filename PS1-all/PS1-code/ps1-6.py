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
from copy import copy



os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS1-all/PS1-images/")


# Load noisy image
img_noisy = cv2.imread("ps1-input2.jpg")
img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)

# Gaussian smoothing

image_noisy_smoothed = cv2.GaussianBlur(img_noisy_gray, (21,21), 5)

# Apply the Canny edge detector on the smoothed noisy image

edges_noisy_smoothed_image = cv2.Canny(image_noisy_smoothed, 30, 50)


# Calculation of the accumulator

H = Hough_Lines(edges_noisy_smoothed_image,1)

# Search for the maximums

threshold = 0.5
max_points = find_max(H,threshold)
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

img_result = cv2.cvtColor(image_noisy_smoothed, cv2.COLOR_GRAY2BGR)
img_result_2 = img_result.copy()

# Draw the edge lines on the image

draw_lines(img_result,d_hough, theta_hough)

# Select the parallels

d_hough_lines, theta_hough_lines = find_parallels(d_hough, theta_hough, 2,30, 100)

# Draw only the parallels


draw_lines(img_result_2,d_hough_lines, theta_hough_lines)

# Several plots

plt.subplot(3,2,1)
plt.imshow(img_noisy_gray, cmap='gray')
plt.draw()


plt.subplot(3,2,2)
plt.imshow(image_noisy_smoothed, cmap='gray')
plt.draw()


plt.subplot(3,2,3)
plt.imshow( edges_noisy_smoothed_image, cmap='gray')
plt.draw()
cv2.imwrite("ps1-6-a-edges.png", edges_noisy_smoothed_image)

plt.subplot(3,2,4)
plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps1-6-a-lines.png", img_result)

plt.subplot(3,2,5)
plt.imshow(cv2.cvtColor(img_result_2, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps1-6-c-lines.png", img_result_2)


plt.show()
