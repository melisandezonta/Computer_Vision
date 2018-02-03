import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
###############################################################################

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS0-all/PS0-images")

# Load the first color image
img1 = cv2.imread('ps0-1-a-1.png')


#Plot the image
plt.figure(1)
plt.imshow(img1);plt.title('Red and blue pixels swapped')
plt.xticks([]), plt.yticks([]) 
plt.show()

