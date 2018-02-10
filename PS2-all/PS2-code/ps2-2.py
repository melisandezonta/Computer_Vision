import os
import numpy as np
import cv2
import matplotlib.pyplot  as plt
from time import time
from Functions import disparity_ssd

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS2-all/PS2-images/")

img_left = cv2.cvtColor(cv2.imread("proj2-pair1-L.png"), cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(cv2.imread("proj2-pair1-R.png"), cv2.COLOR_BGR2GRAY)


plt.subplot(3, 2, 1)
plt.imshow(img_left, cmap="gray")
plt.draw()

plt.subplot(3, 2, 2)
plt.imshow(img_right, cmap="gray")
plt.draw()

start_L = time()
D_L = disparity_ssd(img_left, img_right,100,True)  # left to right
end_L = time()

print("The left to right run took {:.3f} seconds".format(end_L - start_L))

start_R = time()
D_R = disparity_ssd(img_right, img_left, 100, False)  # right to left
end_R = time()

print("The right to left run took {:.3f} seconds".format(end_R - start_R))

#  Map to full 8-bit range
D_L = np.matrix([(d - np.min(D_L)) / (np.max(D_L) - np.min(D_L)) * 255 for d in D_L])
D_R = np.matrix([(d - np.min(D_R)) / (np.max(D_R) - np.min(D_R)) * 255 for d in D_R])

plt.subplot(3, 2, 3)
plt.imshow(D_L, cmap="gray")
plt.draw()
cv2.imwrite("ps2-2-a-lefttoright.png", D_L)

plt.subplot(3, 2, 4)
plt.imshow(D_R, cmap="gray")
plt.draw()
cv2.imwrite("ps2-2-a-righttoleft.png", D_R)

ground_left = cv2.cvtColor(cv2.imread("proj2-pair1-Disp-L.png"), cv2.COLOR_BGR2GRAY)
ground_right = cv2.cvtColor(cv2.imread("proj2-pair1-Disp-R.png"), cv2.COLOR_BGR2GRAY)

plt.subplot(3, 2, 5)
plt.imshow(ground_left, cmap="gray")
plt.draw()
cv2.imwrite("ps2-2-b-ground_truth_left.png", ground_left)

plt.subplot(3, 2, 6)
plt.imshow(ground_left, cmap="gray")
plt.draw()
cv2.imwrite("ps2-2-b-ground_truth_right.png", ground_right)

plt.show()

