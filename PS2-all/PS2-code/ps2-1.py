import os
import numpy as np
import cv2
import matplotlib.pyplot  as plt
from Functions import disparity_ssd

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS2-all/PS2-images/")

img_left  = cv2.cvtColor(cv2.imread("leftTest.png"), cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(cv2.imread("rightTest.png"), cv2.COLOR_BGR2GRAY)

plt.subplot(2,2,1)
plt.imshow(img_left, cmap="gray")
plt.draw()

plt.subplot(2,2,2)
plt.imshow(img_right, cmap="gray")
plt.draw()

D_L = disparity_ssd(img_left,  img_right, 9, 5, 0) # left to right
D_R = disparity_ssd(img_right, img_left,  9, 5, 1) # right to left

D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(2, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-1-a-lefttoright.png", D_L)

plt.subplot(2, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-1-a-righttoleft.png", D_R)

plt.show()

