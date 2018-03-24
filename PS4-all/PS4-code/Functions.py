import numpy as np
import cv2
from math import *
import random
import itertools

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
        min_r = int(r-floor(window_size/2.0))
        max_r = int(r+ceil(window_size/2.0))
        for c in range(int(floor(window_size / 2.0)), int(cols - floor(window_size / 2.0))):

            # Initializing the matrix M

            M = np.zeros([2,2])

            # Compute the dimension of the window in function of the pixel location

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

        #Prepare row slicing
        if w > i[0] or w > (Rt.shape[0] - i[0]):
            wr = min(i[0], Rt.shape[0] - i[0]) * 2
            if wr == 2*i[0]: #Close to upper boundary
                rslice = slice(int(i[0]-floor(wr/2.0)), int(i[0]+ceil(w/2.0)))
            else: # Close to lower boundary
                rslice = slice(int(i[0]-floor(w/2.0)), int(i[0]+ceil(wr/2.0)))
        else:
            rslice = slice(int(i[0]-floor(w/2.0)), int(i[0]+ceil(w/2.0)))

        #Prepare column slicing
        if w > i[1] or w > (Rt.shape[1] - i[1]):
            wc = min(i[1], Rt.shape[1] - i[1]) * 2
            if wc == 2*i[1]: #Close to left boundary
                cslice = slice(int(i[1]-floor(wc/2.0)), int(i[1]+ceil(w/2.0)))
            else: #Close to right boundary
                cslice = slice(int(i[1]-floor(w/2.0)), int(i[1]+ceil(wc/2.0)))
        else:
            cslice = slice(int(i[1]-floor(w/2.0)), int(i[1]+ceil(w/2.0)))

        Rt[rslice, cslice] = 0 #Zero out all values inside window
        corners.append(tuple(i))

    return corners

def corners_markers(corners,image):

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in corners:
        cv2.circle(image_color, c[::-1], 5, (0, 0, 255), 2)
    return image_color

def compute_angle(Ix,Iy): return np.arctan2(Iy, Ix)


def directions(image, corners,angle,l):
    image_col = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in corners:
        [y_a, x_a] = c
        x_b = int(floor(x_a + l * cos(angle[c])))
        y_b = int(floor(y_a + l * sin(angle[c])))
        cv2.arrowedLine(image_col, (x_a, y_a), (x_b, y_b), (0,0,255), 1)
    return image_col



def descriptors_description(image, corners, directions):

    #Keypoints description
    keypoints = []
    for c in corners:
        keypoints.append(cv2.KeyPoint(c[1], c[0], 1, directions[c]))
    #Computation of the descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    _ , descriptors = sift.compute(image, keypoints)

    return keypoints, descriptors

def descriptors_matching(desc1, desc2):

    bfmatcher = cv2.BFMatcher()
    matching_points = bfmatcher.match(desc1, desc2)
    return matching_points


def generate_colors(N, sat, val):
    #Used in order to identify the matches with different colors
    colors = []
    step = int(floor(179/N))
    hsv = np.zeros([1, 1, 3], dtype="uint8")
    hsv[0,0,1] = sat
    hsv[0,0,2] = val
    for n in range(N):
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors.append(tuple([int(bgr[0,0,0]), int(bgr[0,0,1]), int(bgr[0,0,2])]))
        hsv[0,0,0] += step
    return colors

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    image1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    result = np.hstack((image1_color, image2_color))
    cols = image1.shape[1]

    matches = sorted(matches, key = lambda x:x.distance)
    colors = generate_colors(15, 255, 255)

    for i,m in enumerate(matches):
        kp1 = keypoints1[m.queryIdx]
        kp2 = keypoints2[m.trainIdx]
        x1 = int(kp1.pt[0])
        y1 = int(kp1.pt[1])
        x2 = int(kp2.pt[0]) + cols
        y2 = int(kp2.pt[1])
        cv2.line(result, (x1, y1), (x2, y2), colors[i%len(colors)], 1)
    return result

def translational_ransac(matches, keypoints1, keypoints2, tolerance):

    pool = list(matches)
    length = len(pool)
    consensus = {}


    while length:
        #Extract random element from the pool and remove it
        index = random.randrange(length)
        match = pool[index]
        pool[index] = pool[length-1]
        length -= 1

        #Compute translations
        dx = int(keypoints1[match.queryIdx].pt[0]) - int(keypoints2[match.trainIdx].pt[0])
        dy = int(keypoints1[match.queryIdx].pt[1]) - int(keypoints2[match.trainIdx].pt[1])

        #Compute consensus set
        if (dx, dy) not in consensus:
            set = []
            for m in matches:
                #Extract coordinates
                x1, y1 = [int(c) for c in keypoints1[m.queryIdx].pt]
                x2, y2 = [int(c) for c in keypoints2[m.trainIdx].pt]
                #Tests
                Xp = (x1 - x2) < dx + tolerance
                Xn = (x1 - x2) > dx - tolerance
                Yp = (y1 - y2) < dy + tolerance
                Yn = (y1 - y2) > dy - tolerance
                if (Xp and Xn) and (Yp and Yn):
                    set.append(m)
            consensus[(dx, dy)] = set

    #Get biggest consensus set
    translation = max(consensus, key=lambda k:len(consensus[k]))
    return translation, consensus[translation]


def best_match_after_translation(image1, image2, translation):
    rows, cols = image1.shape
    dx, dy = translation
    result = np.zeros([rows + dy, cols + dx, 3], dtype="uint8")
    result[:rows, :cols, :] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    result[dy:(rows+dy), dx:(cols+dx), :] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    return result


def similarity_ransac(matches, keypoints1, keypoints2, tolerance):

    pool = list(itertools.combinations(list(range(len(matches))), 2))
    consensus = {}
    length = len(pool)
    while length:
        # Extract two random elements from the pool and remove it
        index = random.randrange(length)
        match1 = matches[pool[index][0]]
        match2 = matches[pool[index][1]]
        pool[index] = pool[length-1]
        length -= 1

        #Extract left point
        xl1, yl1 = [int(c) for c in keypoints1[match1.queryIdx].pt]
        xl2, yl2 = [int(c) for c in keypoints1[match2.queryIdx].pt]

        #Extract right point
        xr1, yr1 = [int(c) for c in keypoints2[match1.trainIdx].pt]
        xr2, yr2 = [int(c) for c in keypoints2[match2.trainIdx].pt]

        #Compute similarity matrix
        L = np.array([[xl1, -yl1, 1, 0],
                      [yl1, xl1, 0, 1],
                      [xl2, -yl2, 1, 0],
                      [yl2, xl1, 0, 1]])
        R = np.vstack([xr1, yr1, xr2, yr2])
        X,_,_,_ = np.linalg.lstsq(L,R)
        a = X[0,0]; b = X[1,0]; c = X[2,0]; d = X[3,0];
        S = np.array([[a, -b, c],
                      [b, a, d]])

        #Compute consensus set
        if (a, b, c, d) not in consensus:
            set = []
            for m in matches:
                #Extract coordinates
                xl, yl = [int(c) for c in keypoints1[m.queryIdx].pt]
                xr, yr = [int(c) for c in keypoints2[m.trainIdx].pt]
                #Compute resulting point
                P = S.dot(np.vstack([xl, yl, 1]))
                #Tests
                Xerr = abs(xr - P[0,0]) < tolerance
                Yerr = abs(yr - P[1,0]) < tolerance
                if Xerr and Yerr:
                    set.append(m)
            consensus[(a, b, c, d)] = set

    #Get biggest consensus set
    params = max(consensus, key=lambda k:len(consensus[k]))
    similarity = np.array([[params[0], -params[1], params[2]],
                           [params[1], params[0], params[3]]])
    return similarity, consensus[params]


def best_match_after_similirarity(image1, image2, similarity):

    rows, cols = image1.shape
    dx = int(floor(cols/2))
    dy = int(floor(rows/2))

    image1_bgr = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2_bgr = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    result = np.zeros([2*rows, 2*cols, 3], dtype="uint8")

    min_r = result.shape[0]
    max_r = 0
    min_c = result.shape[1]
    max_c = 0
    for r in range(rows):
        for c in range(cols):
            P = similarity.dot(np.vstack([c, r, 1]))
            y = int(P[1,0]) + dy
            x = int(P[0,0]) + dx
            result[y, x, :] = image1_bgr[r,c,:]
            min_r = y if y < min_r else min_r
            max_r = y if y > max_r else max_r
            min_c = x if x < min_c else min_c
            max_c = x if x > max_c else max_c

    result[dy:(rows+dy), dx:(cols+dx), :] = image2_bgr

    return result[min_r:max_r, min_c:max_c, :]
