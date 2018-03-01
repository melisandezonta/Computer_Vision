import os
import numpy as np
from Functions import *
from matplotlib import pyplot as plt

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS3-all/")


image_a = cv2.imread("pic_a.jpg")
image_b = cv2.imread("pic_b.jpg")


array_2D_points_a = extract_2D_points("pts2d-pic_a.txt")
array_2D_points_b = extract_2D_points("pts2d-pic_b.txt")

array_2D_points_homog_a = conv_2_homogeneous_coords(array_2D_points_a)
array_2D_points_homog_b = conv_2_homogeneous_coords(array_2D_points_b)

# Create transform matrices Ta and Tb

Ta = transform_matrix(array_2D_points_a)
Tb = transform_matrix(array_2D_points_b )

# Normalization of the points to create a new fundamental matrix

array_2D_points_homog_a_norm = np.dot(array_2D_points_homog_a ,Ta)
array_2D_points_homog_b_norm = np.dot(array_2D_points_homog_b ,Tb)

F = svd_fundamental(array_2D_points_homog_a_norm, array_2D_points_homog_b_norm)

# Rank reduction using SVD
u, s, v = np.linalg.svd(F)
s[-1] = 0
F_r = u.dot(np.diag(s)).dot(v)

# Creation of a better F

Ff = Tb.T.dot(F_r).dot(Ta)

# Drawing of the epipolar lines
epipolar_lines(image_a, image_b, array_2D_points_homog_b, Ff.T)
epipolar_lines(image_b, image_a, array_2D_points_homog_a, Ff)

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps3-2-5-a.jpg", image_a)

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
plt.draw()
cv2.imwrite("ps3-2-5-b.jpg", image_b)

plt.show()