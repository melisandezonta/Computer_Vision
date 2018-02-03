import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
###############################################################################

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS0-all/PS0-images")

# Load the two color images

img1 = cv2.imread('mandrill.png')
img2 = cv2.imread('lake.png')



#OpenCV follows BGR order, while matplotlib likely follows RGB order.
#So when you display an image loaded in OpenCV using pylab functions, 
#you need to convert it into RGB mode. 

img1b =  img1[:,:,::-1]
img2b = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#Plot the 2 images
plt.figure(1)
plt.subplot(121); plt.imshow(img1b);plt.title('Mandrill')

plt.xticks([]), plt.yticks([]) 
plt.subplot(122); plt.imshow(img2b);plt.title('Sailboat')
plt.xticks([]), plt.yticks([]) 
plt.show()

