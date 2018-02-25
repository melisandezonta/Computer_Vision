import os
from Functions import *


os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS3-all/")

M_normA = np.array([[-0.4583, 0.2947, 0.0139, -0.0040],
                     [0.0509, 0.0546, 0.5410, 0.0524],
                     [-0.1090, -0.1784, 0.0443, -0.5968]], dtype=np.float32)

M = np.loadtxt("m.txt")

C = camera_center(M)

print("The location of the camera in the 3D world is {}".format(C))
