import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
###############################################################################

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS0-all/PS0-images")



# Load the two images 
img1 = cv2.imread('ps0-1-a-1.png')

# Green monochrome version of the image 1

b1,g1,r1 = cv2.split(img1)
M1g = g1

# Calculation of the statistics parameters of the green monochrome image 1

mean_g = M1g.mean()
std_g = M1g.std()

# Construction of the new image
M1g_new_1 = ((M1g - mean_g)/std_g) 
M1g_new_2 = M1g_new_1* 10 
M1g_new_3 = M1g_new_2 + mean_g

#Visualize the effect of the transformation

cv2.imshow('Original monochrome image',M1g)
cv2.imshow('Normalisation',M1g_new_1)
cv2.imshow('Second step of the transformation',M1g_new_2)
cv2.imshow('Final result',M1g_new_3)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('ps0-4-b-1.png', M1g.astype(np.uint8))
    cv2.imwrite('ps0-4-b-2.png', M1g_new_1.astype(np.uint8))
    cv2.imwrite('ps0-4-b-3.png', M1g_new_2.astype(np.uint8))
    cv2.imwrite('ps0-4-b-4.png', M1g_new_3.astype(np.uint8))
    cv2.destroyAllWindows()