import os
import numpy as np
from Functions import *


os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS3-all/")

array_2D_points = extract_2D_points("pts2d-norm-pic_a.txt")
array_3D_points = extract_3D_points("pts3d-norm.txt")


array_2D_points_homog = conv_2_homogeneous_coords(array_2D_points)
array_3D_points_homog = conv_2_homogeneous_coords(array_3D_points)


M,res = svd_solver(array_2D_points_homog, array_3D_points_homog)

y = M.dot(array_3D_points_homog[-1, :])
y = y/y[-1]

print("The resulting projection matrix is \n\n {} \n\n".format(M))
print("The last point is \n\n {} \n\n".format(y))
print("The residual is {:.4f}".format(res[-1]))

