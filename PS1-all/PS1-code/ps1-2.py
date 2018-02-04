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

img = cv2.imread('ps1-input0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,100,200)

# Calculation of the accumulator

H = Hough_Lines(edges,1)

# Visualisation of the accumulator

accumulator = cv2.cvtColor(H, cv2.COLOR_GRAY2BGR)

# Search for the maximums

threshold = 0.5
max_points = find_max(H,threshold)
number_max_points = len(max_points[1])

# Move back in the polar domain


[m,n] = edges.shape
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

# Move back to a RGB image so we can visualize the green lines on it

img1 = cv2.imread('ps1-input0.png')


# Draw the edge lines on the image

draw_lines(img1,d_hough, theta_hough)

# Diverse plots


plt.subplot(2,2,1)
plt.imshow(edges,cmap = 'gray')
plt.draw()
cv2.imwrite("ps1-2-edges.png",edges)

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(accumulator, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps1-2-accu-circled.png",accumulator)

plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps1-2-lines.png", img1)

plt.show()


