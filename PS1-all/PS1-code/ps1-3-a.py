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

gaussian_kernel = cv2.getGaussianKernel(11, 1.7)

# Apply the gaussian filter to the image

image_noisy_smoothed = cv2.sepFilter2D(img_noisy, -1, gaussian_kernel, gaussian_kernel)

plt.imshow(image_noisy_smoothed, cmap='gray')
plt.draw()
cv2.imwrite("ps1-3-a-image-noisy-smoothed.png", image_noisy_smoothed)


plt.show()

