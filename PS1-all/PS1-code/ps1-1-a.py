#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:13:17 2018

@author: melisandezonta
"""
import os
import cv2
from matplotlib import pyplot as plt

os.chdir("/Users/melisandezonta/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS1-all/PS1-images/")


img = cv2.imread('ps1-input0.png',0)
edges = cv2.Canny(img,100,200)
            
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
cv2.imwrite("ps1-1-a-original.png",img)
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
cv2.imwrite("ps1-1-a-edges.png",edges)
plt.show()
