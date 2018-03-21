import numpy as np
import cv2
from math import *


def compute_gradients(image,aperture_size):

    # Gaussian derivative kernels
    g = cv2.getGaussianKernel(aperture_size, -1) * cv2.getGaussianKernel(aperture_size, -1).T
    gy, gx = np.gradient(g)

    # Filter with the kernel computed below and apply to the image and retrieve gradients
    Ix = cv2.filter2D(image, cv2.CV_32F, gx)
    Iy = cv2.filter2D(image, cv2.CV_32F, gy)

    return Ix,Iy

def colors_scaling(gradient): return np.asarray((gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient)) * 255, dtype="uint8")

def Harris_corner_response(Ix,Iy,alpha,window):

    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = np.multiply(Ix,Iy)

    rows,cols = Ix.shape
    window_size = window.shape[0]

    R = np.zeros([rows,cols])

    for r in range(int(floor(window_size / 2.0)), int(rows - floor(window_size / 2.0))):
        for c in range(int(floor(window_size / 2.0)), int(cols - floor(window_size / 2.0))):

            # Initializing the matrix M

            M = np.zeros([2,2])

            # Compute the dimension of the window in function of the pixel location

            min_r = int(r-floor(window_size/2.0))
            max_r = int(r+ceil(window_size/2.0))
            min_c = int(c-floor(window_size/2.0))
            max_c = int(c+ceil(window_size/2.0))

            # Determining the matrix M values

            M[0, 0] = np.sum(np.multiply(window, Ix2[min_r:max_r, min_c:max_c]))
            M[1, 0] = np.sum(np.multiply(window, Ixy[min_r:max_r, min_c:max_c]))
            M[0, 1] = np.sum(np.multiply(window, Ixy[min_r:max_r, min_c:max_c]))
            M[1, 1] = np.sum(np.multiply(window, Iy2[min_r:max_r, min_c:max_c]))


            # Compute the value of R for the matrix M at pixel location [r,c]

            R[r,c] = np.linalg.det(M) - alpha*np.trace(M)

    return R


def Harris_thresholding(R,threshold):
    R[R < threshold] = 0
    return R

def non_maxima_suppression(Rt,w):
    corners = []
    while np.argwhere(Rt > 0).shape[0]:
        i = np.unravel_index(np.argmax(Rt), Rt.shape) # Find current maximum

        # Prepare row slicing
        if w > i[0] or w > (Rt.shape[0] - i[0]):
            wr = min(i[0], Rt.shape[0] - i[0]) * 2
            if wr == 2*i[0]: # Close to upper boundary
                rslice = slice(int(i[0]-floor(wr/2.0)), int(i[0]+ceil(w/2.0)))
            else: # Close to lower boundary
                rslice = slice(int(i[0]-floor(w/2.0)), int(i[0]+ceil(wr/2.0)))
        else:
            rslice = slice(int(i[0]-floor(w/2.0)), int(i[0]+ceil(w/2.0)))

        # Prepare column slicing
        if w > i[1] or w > (Rt.shape[1] - i[1]):
            wc = min(i[1], Rt.shape[1] - i[1]) * 2
            if wc == 2*i[1]: # Close to left boundary
                cslice = slice(int(i[1]-floor(wc/2.0)), int(i[1]+ceil(w/2.0)))
            else: # Close to right boundary
                cslice = slice(int(i[1]-floor(w/2.0)), int(i[1]+ceil(wc/2.0)))
        else:
            cslice = slice(int(i[1]-floor(w/2.0)), int(i[1]+ceil(w/2.0)))

        Rt[rslice, cslice] = 0 # Zero out all values inside window
        corners.append(tuple(i))

    return corners

def corners_markers(corners,image):

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in corners:
        cv2.circle(image_color, c[::-1], 5, (0, 0, 255), 2)
    return image_color

