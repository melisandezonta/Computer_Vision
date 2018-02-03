import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
###############################################################################


os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS0-all/PS0-images")



# Load the two images 
img1 = cv2.imread('ps0-1-a-1.png')

# Green monochrome version of the image 1

b,g,r = cv2.split(img1)
M1g = g

# Shift the image

m1,n1 = M1g.shape
M = np.float32([[1,0,-2],[0,1,0]])
M1g_shifted = cv2.warpAffine(M1g,M,(m1,n1))

#Plot the image

cv2.imshow('Original monochrome image',M1g)
cv2.imshow('Shifted image',M1g_shifted)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('ps0-4-c-1.png',M1g)
    cv2.imwrite('ps0-4-c-2.png',M1g_shifted)
    cv2.destroyAllWindows()