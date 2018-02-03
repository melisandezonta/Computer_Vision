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

gaussian_kernel = cv2.getGaussianKernel(13, 2.5)
image_noisy_smoothed = cv2.sepFilter2D(img_noisy, -1, gaussian_kernel, gaussian_kernel)

# Apply the Canny edge detector on the smoothed noisy image

edges_noisy_smoothed_image = cv2.Canny(image_noisy_smoothed, 60, 100)

plt.subplot(1,2,1)
plt.imshow(edges_noisy_smoothed_image, cmap='gray')
plt.draw()
cv2.imwrite("ps1-3-b-edges-image-noisy.png", edges_noisy_smoothed_image)

# output the edges detected with Canny on the original image

img_edges = cv2.Canny(img_gray, 50, 150)
plt.subplot(1,2,2)
plt.imshow(img_edges, cmap='gray')
plt.draw()
cv2.imwrite("ps1-3-b-edges.png", img_edges)

plt.show()
