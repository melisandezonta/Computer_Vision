import os
import numpy as np
from Functions import *


os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS3-all/")

k = [8,12,16]


array_2D_points = extract_2D_points("pts2d-pic_a.txt")
array_3D_points = extract_3D_points("pts3d.txt")

array_2D_points_homog = conv_2_homogeneous_coords(array_2D_points)
array_3D_points_homog = conv_2_homogeneous_coords(array_3D_points)


k = [8, 12, 16]
number_repetitions = 10


residuals_average = []
res_average_mins = []
M_mins = []

for i in k:
    residuals_average = np.zeros(number_repetitions)
    for n in range(number_repetitions):

        # Randomly choose k points from the 2D list and their corresponding points in the 3D list

        random_points = np.random.choice(array_3D_points_homog.shape[0], i, False)

        array_2D_points_homog_random = array_2D_points_homog[random_points,:]
        array_3D_points_homog_random = array_3D_points_homog[random_points,:]

        # Compute the projection matrix M on the chosen points

        M, res = svd_solver(array_2D_points_homog_random, array_3D_points_homog_random)

        # Select all the points not in the set of k

        array_2D_points_homog_rand_four = np.delete(array_2D_points_homog.copy(), random_points, 0)
        array_3D_points_homog_rand_four = np.delete(array_3D_points_homog.copy(), random_points, 0)

        # Pick 4 points

        four_points = np.random.choice(array_3D_points_homog_rand_four.shape[0], 4, False)

        # Compute the average residual
        residuals_average[n] = np.mean(residuals_calculation(array_2D_points_homog_rand_four[four_points,:],
                                                             array_3D_points_homog_rand_four[four_points,:], M))

        res_avg_min = residuals_average[n]
        M_min = M

    res_average_mins.append(res_avg_min)
    M_mins.append(M_min)

    print("The average residuals for k = {} are \n".format(i))
    print(residuals_average, "\n")
    print("The projection matrix for the lowest residual (R = {:.4f}) is \n\n {} \n\n==========\n".format(res_avg_min, M_min))

min_i = np.argmin(res_average_mins)
print("The best projection matrix overall (k = {}) is \n\n {}".format(k[min_i], M_mins[min_i]))

np.savetxt("m.txt", M_mins[min_i])
