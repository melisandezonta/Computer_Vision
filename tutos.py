#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 13:29:29 2018

@author: melisandezonta
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
###############################################################################


os.chdir('/Users/melisandezonta/Desktop/')
## Basic Operations on Images


img = cv2.imread('messi5.jpg')

px = img[100,100]
print(px)

# accessing only blue pixel
blue = img[100,100,0]
print(blue)

img[100,100] = [255,255,255]
print(img[100,100])

# accessing RED value
img.item(10,10,2)

# modifying RED value
img.itemset((10,10,2),100)
img.item(10,10,2)

print('The image shape is : ', img.shape)
print('The image size is : ', img.size)

print('The image type is : ',img.dtype)


#Checker cette partie du code (ne fonctionne pas tel quel)
ball = img[280:340, 330:390]
#img[273:333, 100:160] = ball

b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))



BLUE = [255,0,0]

img1 = cv2.imread('opencv_logo.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()

###############################################################################

## Arithmetic Operations on Images

x = np.uint8([250])
y = np.uint8([10])

print(cv2.add(x,y)) # 250+10 = 260 => 255


print(x+y)

img1 = cv2.imread('ml.png')
img2 = cv2.imread('opencv_logo.png')

dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()