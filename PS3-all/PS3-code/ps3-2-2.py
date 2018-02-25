import os
import numpy as np
from Functions import *


os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS3-all/")

array_2D_points_a = extract_2D_points("pts2d-pic_a.txt")
array_2D_points_b = extract_2D_points("pts2d-pic_b.txt")


array_2D_points_homog_a = conv_2_homogeneous_coords(array_2D_points_a)
array_2D_points_homog_b = conv_2_homogeneous_coords(array_2D_points_b)

F = svd_fundamental(array_2D_points_homog_a, array_2D_points_homog_b)

# Rank reduction using SVD
u, s, v = np.linalg.svd(F)
s[-1] = 0
F_r = u.dot(np.diag(s)).dot(v)

print("The resulting fundamental matrix is \n\n {} \n\n".format(F_r))
