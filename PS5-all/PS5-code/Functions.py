from math import *
import numpy as np
import cv2

def power2(x):
    return x != 0 and ((x & (x - 1)) == 0)

def next_power2(x):
    while(not power2(x)):
        x += 1
    return x

def zero_padding(image):
    rows, cols = image.shape
    rows_new = rows; cols_new = cols;
    if not power2(rows):
        rows_new = next_power2(rows)
    if not power2(cols):
        cols_new = next_power2(cols)
    result = np.zeros([rows_new, cols_new])
    result[:rows, :cols] = image
    return result, rows_new - rows, cols_new - cols

def reduce(level0, a):

    # Kernel building
    w = np.array([[1/4-a/2, 1/4, a, 1/4, 1/4-a/2]])
    kernel = w * w.T

    m, n = level0.shape
    level1 = np.zeros([int(m/2), int(n/2)], dtype="float32")

    # Mirrors an image borders
    G = cv2.copyMakeBorder(level0, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    for r,i in zip(list(range(2, m+2, 2)), list(range(int(m/2)))):
        for c,j in zip(list(range(2, n+2, 2)), list(range(int(n/2)))):
            min_r = int(r - 2); max_r = int(r + 3)
            min_c = int(c - 2); max_c = int(c + 3)
            level1[i,j] = np.sum(np.multiply(kernel, G[min_r:max_r, min_c:max_c]))

    return level1

def remove_zero_padding(image, level, rows_diff, cols_diff):
    m, n = image.shape
    R = int(floor(rows_diff / pow(2, level)))
    C = int(floor(cols_diff / pow(2, level)))
    r_index = slice(0, m - R) if R > 0 else slice(0, m)
    c_index = slice(0, n - C) if C > 0 else slice(0, n)
    return image[r_index, c_index]

def Gaussian_Pyramid(image, n, cut=False):

    pyramid = []
    zero_padded_image, diff_row, diff_cols = zero_padding(image)
    pyramid.append(zero_padded_image)
    for i in range(n):
        level = reduce(pyramid[i], 0.4) # for a = 0.4 it is Gaussian-like
        pyramid.append(level)
        if cut:
            pyramid = [remove_zero_padding(lev, i, diff_row, diff_cols) for i, lev in enumerate(pyramid)]
    return pyramid

def expand(level0, a):
    # Kernel building
    w = np.array([[1/4-a/2, 1/4, a, 1/4, 1/4-a/2]])
    kernel = w * w.T

    m, n = level0.shape
    level1 = np.zeros([int(m*2), int(n*2)], dtype="float32")

    # Mirrors an image borders
    G = cv2.copyMakeBorder(level0, 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    for i in range(int(m*2)):
        for j in range(int(n*2)):
            level1[i,j] = 0
            M = [-2,0,2] if i % 2 == 0 else [-1,1]
            N = [-2,0,2] if j % 2 == 0 else [-1,1]
            for m in M:
                for n in N:
                    r = int((i-m)/2); c = int((j-n)/2);
                    level1[i,j] += kernel[m+2,n+2] * G[r+2,c+2]
            level1[i,j] *= 4

    return level1

def scale_values(X): return np.asarray((X - np.min(X)) / (np.max(X) - np.min(X)) * 255, dtype="uint8")


def Lucas_Kanade(left, right, size, sigma):

    rows, cols = left.shape

    # Smooth images
    kernel = cv2.getGaussianKernel(size, sigma) * cv2.getGaussianKernel(size, sigma).T
    left = cv2.filter2D(left, -1, kernel)
    right = cv2.filter2D(right, -1, kernel)

    # Compute gradients
    g = np.array([[-1, 0, 1]])
    Ix1 = cv2.filter2D(left, cv2.CV_32F, g)
    Iy1 = cv2.filter2D(left, cv2.CV_32F, g.T)

    It = np.asarray(right, dtype="float32") - np.asarray(left, dtype="float32")

    Ixx = np.multiply(Ix1, Ix1)
    Iyy = np.multiply(Iy1, Iy1)
    Ixy = np.multiply(Ix1, Iy1)
    Ixt = np.multiply(Ix1, It)
    Iyt = np.multiply(Iy1, It)

    # Prepare weighting window
    w = np.ones([size, size])

    # Compute weighted sums
    Ixx = cv2.filter2D(Ixx, -1, w)
    Iyy = cv2.filter2D(Iyy, -1, w)
    Ixy = cv2.filter2D(Ixy, -1, w)
    Ixt = cv2.filter2D(Ixt, -1, w)
    Iyt = cv2.filter2D(Iyt, -1, w)

    U = np.zeros([rows, cols], dtype="float32")
    V = np.zeros([rows, cols], dtype="float32")
    for r in range(rows):
        for c in range(cols):
            L = np.array([[Ixx[r,c], Ixy[r,c]],[Ixy[r,c], Iyy[r,c]]])
            R = np.array([[-Ixt[r,c]], [-Iyt[r,c]]])
            if np.linalg.det(L) < 1e-8:
                U[r,c] = 0
                V[r,c] = 0
            else:
                X = np.linalg.inv(L).dot(R)
                U[r,c] = X[0,0]
                V[r,c] = X[1,0]
    return cv2.medianBlur(U, 5), cv2.medianBlur(V, 5)


def warp(image, U, V):
    m,n = image.shape
    X, Y = np.meshgrid(np.arange(n), np.arange(m), indexing="xy")
    Xmap = np.asarray(X - U, dtype="float32")
    Ymap = np.asarray(Y - V, dtype="float32")
    warped_nearest = cv2.remap(image, Xmap, Ymap, cv2.INTER_NEAREST)
    warped_linear = cv2.remap(image, Xmap, Ymap, cv2.INTER_LINEAR)
    indices = np.where(warped_linear == 0)
    warped_linear[indices] = warped_nearest[indices]
    return warped_linear


def Hierarchical_LK(left, right, n, a, size, sigma):

    # Initialize k = n where n is the max level
    pyr_left  = Gaussian_Pyramid(left, n, cut=True)
    pyr_right = Gaussian_Pyramid(right, n, cut=True)

    for k in range(n, -1, -1):

        # reduce both input images to level k
        Lk = pyr_left[k]
        Rk = pyr_right[k]

        # If k = n initialize U and V to be zero inages the size of Lk;
        if k == n:
            U = np.zeros(Lk.shape, dtype="float32")
            V = np.zeros(Lk.shape, dtype="float32")
        # otherwise expand the flow field and double to get the next level
        else:
            U = 2 * expand(U, a)
            V = 2 * expand(V, a)

        # Reduce U and V with zeros if size different than images
        if U.shape[0] != Lk.shape[0] or U.shape[1] != Lk.shape[1]:

            R = U.shape[0] != Lk.shape[0]
            C = U.shape[1] != Lk.shape[1]
            U = U[:-R, :-C]
            V = V[:-R, :-C]

        # Warp Lk using U and V to form Wk
        Wk = warp(Lk, U, V)

        # Perform LK on Wk and Rk to yield two incremental flow fields Dx and Dy
        Dx, Dy = Lucas_Kanade(np.float32(Wk), np.float32(Rk), size, sigma)

        # Add these to the original flow
        U = U + Dx
        V = V + Dy

    return U, V, Wk, Lk, Rk


