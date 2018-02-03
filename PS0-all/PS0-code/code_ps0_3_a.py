import copy
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
###############################################################################

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS0-all/PS0-images")

# Load the two images 
img1 = cv2.imread('ps0-1-a-1.png')
img2 = cv2.imread('ps0-1-a-2.png')

# Create the monochromes images
# Select red channel of each image

b1,g1,r1 = cv2.split(img1)
b2,g2,r2 = cv2.split(img2)

M1r = r1
M2r = r2

# Size of the images

[m1,n1,l1] = img1.shape
[m2,n2,l2] = img2.shape

# Portion of size 100*100 pixels for each image


square = M1r[int(m1/2)-50:int(m1/2)+50,int(n1/2)-50:int(n1/2)+50]
r2_m = copy.copy(r2)
r2_m[int(m2/2)-50:int(m2/2)+50,int(n2/2)-50:int(n2/2)+50] = square

#Visualize the insertion


cv2.imshow('Original Image 1',img1)
cv2.imshow('Original Image 2',img2)
cv2.imshow('Red Monochrome Image 1',M1r)
cv2.imshow('Red Monochrome Image 2',M2r)
cv2.imshow('Inserted image 1 ROI into image2',r2_m)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('ps0-3-a-1.png',img1)
    cv2.imwrite('ps0-3-a-2.png',img2)
    cv2.imwrite('ps0-3-a-3.png',M1r)
    cv2.imwrite('ps0-3-a-4.png',M2r)
    cv2.imwrite('ps0-3-a-5.png',r2_m)
    cv2.destroyAllWindows()