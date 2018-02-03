import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
###############################################################################

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS0-all/PS0-images")

# Load the first color image
img1 = cv2.imread('ps0-1-a-1.png')

b,g,r = cv2.split(img1)
M1r = r

#Plot the image

cv2.imshow('Monochrome image after selection of the red channel',M1r)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('ps0-2-c.png',M1r)
    cv2.destroyAllWindows()