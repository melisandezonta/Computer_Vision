import os
import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt
###############################################################################

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS0-all/PS0-images")

# Load the image 
img1 = cv2.imread('ps0-1-a-1.png')
#img1 = cv2.imread('/Users/melisandezonta/Desktop/essai.png')

# BLue monochrome version of the image 1
b,g,r = cv2.split(img1)
M1b = b

# Creation of the noisy blue channel
m1,n1= M1b.shape
mu = 0
var = 1000
sigma = var**0.5
gaussian = np.random.normal(mu,sigma,(m1,n1))
gaussian_noise = gaussian.reshape(m1,n1)
M1b_noisy = M1b + gaussian_noise


# Resulting image
img1_noisy = copy.copy(img1)
img1_noisy[:,:,0] = M1b_noisy

#Plot the image

cv2.imshow('Original image',img1)
cv2.imshow('Noisy blue image',img1_noisy)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('ps0-5-b-1.png',img1)
    cv2.imwrite('ps0-5-b-2.png',img1_noisy)
    cv2.destroyAllWindows()