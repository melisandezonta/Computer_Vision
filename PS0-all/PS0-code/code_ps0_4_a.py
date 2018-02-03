import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
###############################################################################

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS0-all/PS0-images")



# Load the two images 
img1 = cv2.imread('ps0-1-a-1.png')

# Green monochrome version of the image 1
img1 =  img1[:,:,::-1]
M1g = img1[:,:,1]

# Calculation of the min and max of the green monochrome image 1
min_g = M1g.min()
max_g = M1g.max()

print('The minimal pixel value of the green monochrome version of the image 1 is : ', min_g )
print('The maximal pixel value of the green monochrome version of the image 1 is : ', max_g )

# Calculation of thestatistics parameters of the green monochrome image 1

mean_g = M1g.mean()
std_g = M1g.std()

print('The mean of the green monochrome version of the image 1 is : ', mean_g )
print('The standard deviation deviation of the green monochrome version of the image 1 is : ', std_g )
