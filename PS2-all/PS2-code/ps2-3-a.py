import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
from Functions import disparity_ssd

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS2-all/PS2-images/")

img_left  = cv2.cvtColor(cv2.imread("proj2-pair1-L.png"), cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(cv2.imread("proj2-pair1-R.png"), cv2.COLOR_BGR2GRAY)

# Add of the noise to the images

image_left_noisy = img_left.copy() + cv2.randn(np.ndarray(img_left.shape, dtype="uint8"), 0, 25)
image_right_noisy = img_right.copy() + cv2.randn(np.ndarray(img_right.shape, dtype="uint8"), 0, 25)


plt.subplot(3, 2, 1)
plt.imshow(img_left, cmap="gray")
plt.draw()

plt.subplot(3, 2, 2)
plt.imshow(img_right, cmap="gray")
plt.draw()

left_image_start = time()
D_L = disparity_ssd(image_left_noisy, image_right_noisy,9,100,0)  # left to right
left_image_end = time()

print("The left to right run took {:.3f} mins".format((left_image_end - left_image_start)/60))

right_image_start  = time()
D_R = disparity_ssd(image_right_noisy, image_left_noisy,9,100,1)  # right to left
right_image_end = time()

print("The right to left run took {:.3f} mins".format((right_image_end - right_image_start)/60))

D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.subplot(3, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-3-a-lefttoright.png", D_L)

plt.subplot(3, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-3-a-righttoleft.png", D_R)

ground_left = cv2.cvtColor(cv2.imread("proj2-pair1-Disp-L.png"), cv2.COLOR_BGR2GRAY)
ground_right = cv2.cvtColor(cv2.imread("proj2-pair1-Disp-R.png"), cv2.COLOR_BGR2GRAY)

plt.subplot(3, 2, 5)
plt.imshow(ground_left, cmap="gray")
plt.draw()
cv2.imwrite("ps2-3-b-ground_truth_left.png", ground_left)

plt.subplot(3, 2, 6)
plt.imshow(ground_left, cmap="gray")
plt.draw()
cv2.imwrite("ps2-3-b-ground_truth_right.png", ground_right)

plt.show()




