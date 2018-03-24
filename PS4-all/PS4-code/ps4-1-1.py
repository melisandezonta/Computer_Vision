import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from Functions import compute_gradients, colors_scaling




os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS4-all/PS4-input/")

image_a = cv2.cvtColor(cv2.imread("transA.jpg"), cv2.COLOR_BGR2GRAY)
m_a, n_a = image_a.shape

image_b = cv2.cvtColor(cv2.imread("simA.jpg"), cv2.COLOR_BGR2GRAY)
m_b, n_b = image_b.shape


# Computing X and Y gradient
Ix_a, Iy_a = compute_gradients(image_a, 3)
Ix_b, Iy_b = compute_gradients(image_b, 3)

# Scaling the output values of the gradients

Ix_a = colors_scaling(Ix_a)
Iy_a = colors_scaling(Iy_a)

Ix_b = colors_scaling(Ix_b)
Iy_b = colors_scaling(Iy_b)

# Concatenate the X gradient and Y gradient

I_gradient_a = np.hstack((Ix_a,Iy_a))
I_gradient_b = np.hstack((Ix_b,Iy_b))


# Diverse plots

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(image_a, cmap="gray")
plt.draw()

plt.subplot(2, 2, 2)
plt.imshow(I_gradient_a, cmap="gray")
plt.draw()
cv2.imwrite("ps4-1-1-transA.png", I_gradient_a)

plt.subplot(2, 2, 3)
plt.imshow(image_b, cmap="gray")
plt.draw()

plt.subplot(2, 2, 4)
plt.imshow(I_gradient_b, cmap="gray")
plt.draw()
cv2.imwrite("ps4-1-1-simA.png", I_gradient_b)

plt.show()




