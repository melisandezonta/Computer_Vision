import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS2-all/PS2-images/")

img_left  = cv2.cvtColor(cv2.imread("proj2-pair2-L.png"), cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(cv2.imread("proj2-pair2-R.png"), cv2.COLOR_BGR2GRAY)

#############################################################################################
# SSD Disparity : Original Test
#############################################################################################
plt.figure(figsize=(8,8))

plt.subplot(3, 2, 1)
plt.imshow(img_left, cmap="gray")
plt.draw()

plt.subplot(3, 2, 2)
plt.imshow(img_right, cmap="gray")
plt.draw()


D_L = disparity_ssd(img_left, img_right,9,100,0)  # left to right
D_R = disparity_ssd(img_right, img_left,9,100,1)  # right to left


D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(3, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-original-ssd-lefttoright.png", D_L)

plt.subplot(3, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-original-ssd-righttoleft.png", D_R)

ground_left = cv2.cvtColor(cv2.imread("proj2-pair2-Disp-L.png"), cv2.COLOR_BGR2GRAY)
ground_right = cv2.cvtColor(cv2.imread("proj2-pair2-Disp-R.png"), cv2.COLOR_BGR2GRAY)

plt.subplot(3, 2, 5)
plt.imshow(ground_left, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-ground_truth_left.png", ground_left)

plt.subplot(3, 2, 6)
plt.imshow(ground_left, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-ground_truth_right.png", ground_right)

plt.show()

##############################################################################################
# SSD Disparity : Add of the gaussian noise
##############################################################################################


image_left_noisy = img_left.copy() + cv2.randn(np.ndarray(img_left.shape, dtype="uint8"), 0, 20)
image_right_noisy = img_right.copy() + cv2.randn(np.ndarray(img_right.shape, dtype="uint8"), 0, 20)


plt.figure(figsize=(8,8))
plt.subplot(2, 2, 1)
plt.imshow(image_left_noisy, cmap="gray")
plt.draw()

plt.subplot(2, 2, 2)
plt.imshow(image_right_noisy, cmap="gray")
plt.draw()


D_L = disparity_ssd(image_left_noisy, image_right_noisy,9,100,0)  # left to right
D_R = disparity_ssd(image_right_noisy, image_left_noisy,9,100,1)  # right to left


D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(2, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-noise-ssd-lefttoright.png", D_L)

plt.subplot(2, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-noise-ssd-righttoleft.png", D_R)

plt.show()


##############################################################################################
# SSD Disparity : Contrast Increase Test
##############################################################################################


mult = np.ndarray(img_right.shape, dtype="float32")
mult.fill(1.1)
image_right_contrasted = cv2.multiply(np.asarray(img_right, dtype="float32"), mult)
image_right_contrasted = np.asarray(image_right_contrasted, dtype="uint8")

plt.figure(figsize=(8,8))
plt.subplot(2, 2, 1)
plt.imshow(img_left, cmap="gray")
plt.draw()

plt.subplot(2, 2, 2)
plt.imshow(image_right_contrasted, cmap="gray")
plt.draw()

D_L = disparity_ssd(img_left, image_right_contrasted,9,100,0)  # left to right
D_R = disparity_ssd(image_right_contrasted, img_left,9,100,1)  # right to left

D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(2, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-contrast-ssd-lefttoright.png", D_L)

plt.subplot(2, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-contrast-ssd-righttoleft.png", D_R)

plt.show()


###################################################################################################
###################################################################################################
##################################################################################################


#############################################################################################
# Cross Correlation : Original Test
#############################################################################################
plt.figure(figsize=(8,8))

plt.subplot(2, 2, 1)
plt.imshow(img_left, cmap="gray")
plt.draw()

plt.subplot(2, 2, 2)
plt.imshow(img_right, cmap="gray")
plt.draw()


D_L = disparity_correlation(img_left, img_right,9,0)  # left to right
D_R = disparity_correlation(img_right, img_left,9,1)  # right to left


D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(2, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-original-corr-lefttoright.png", D_L)

plt.subplot(2, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-original-corr-righttoleft.png", D_R)

plt.show()

##############################################################################################
# Cross Correlation : Add of the gaussian noise
##############################################################################################


image_left_noisy = img_left.copy() + cv2.randn(np.ndarray(img_left.shape, dtype="uint8"), 0, 20)
image_right_noisy = img_right.copy() + cv2.randn(np.ndarray(img_right.shape, dtype="uint8"), 0, 20)


plt.figure(figsize=(8,8))
plt.subplot(2, 2, 1)
plt.imshow(image_left_noisy, cmap="gray")
plt.draw()

plt.subplot(2, 2, 2)
plt.imshow(image_right_noisy, cmap="gray")
plt.draw()


D_L = disparity_correlation(image_left_noisy, image_right_noisy,9,0)  # left to right
D_R = disparity_correlation(image_right_noisy, image_left_noisy,9,1)  # right to left


D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(2, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-noise-corr-lefttoright.png", D_L)

plt.subplot(2, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-noise-corr-righttoleft.png", D_R)

plt.show()


##############################################################################################
# Cross Correlation : Contrast Increase Test
##############################################################################################


mult = np.ndarray(img_right.shape, dtype="float32")
mult.fill(1.1)
image_right_contrasted = cv2.multiply(np.asarray(img_right, dtype="float32"), mult)
image_right_contrasted = np.asarray(image_right_contrasted, dtype="uint8")


plt.subplot(2, 2, 1)
plt.imshow(img_left, cmap="gray")
plt.draw()

plt.subplot(2, 2, 2)
plt.imshow(image_right_contrasted, cmap="gray")
plt.draw()

D_L = disparity_correlation(img_left, image_right_contrasted,9,0)  # left to right
D_R = disparity_correlation(image_right_contrasted, img_left,9,1)  # right to left

D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(2, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-contrast-corr-lefttoright.png", D_L)

plt.subplot(2, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-5-contrast-corr-righttoleft.png", D_R)

plt.show()



