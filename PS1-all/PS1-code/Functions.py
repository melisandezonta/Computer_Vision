#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:21:47 2018

@author: melisandezonta
"""

import cv2
from numpy import *
from math import *


def Hough_to_index(d_theta, d):
    d_idx = ((d_theta - d[0]) / ((d[-1]) - d[0])) * len(d)

    return d_idx


def index_to_Hough(i, d):
    d_hough = d[0] + i * (d[-1] - d[0]) / len(d)

    return d_hough


def Hough_Lines(edges, grid_size):
    [m, n] = edges.shape
    theta = range(-90, 91, grid_size)
    d_max = sqrt((m - 1) ** 2 + (n - 1) ** 2)
    d = range(-ceil(d_max), ceil(d_max) + 1, 1)

    H = zeros((len(d), len(theta)), dtype="uint8")

    for x in range(0, m):

        for y in range(0, n):

            if (edges[x, y] == 255):

                for theta_idx in theta:
                    d_theta = x * cos(deg2rad(theta[theta_idx])) - y * sin(deg2rad(theta[theta_idx]))

                    d_theta_idx = Hough_to_index(d_theta, d)

                    H[int(d_theta_idx), theta_idx] += 1

    return H


def find_max(H, threshold):
    max_H = H.max()
    threshold_max = threshold * max_H
    print(threshold_max)
    inds = where(H > threshold_max)

    return inds


def draw_lines(img, d, theta):
    for i in range(0, len(d)):
        # print(d[i],theta[i])
        a = cos(deg2rad(theta[i]))
        b = sin(deg2rad(theta[i]))
        x0 = a * d[i]
        y0 = b * d[i]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def Hough_Circles(edges, r_min, r_max, grid_size, a_min, a_max, b_min, b_max, a_len, b_len, theta_test):
    [m, n] = edges.shape
    r_len = r_max - r_min
    # Initialize accumulator
    H = zeros((a_len, b_len, r_len), dtype="uint8")

    for y in range(0, m):
        for x in range(0, n):
            if (edges[y, x] == 255):
                for r in range(r_min, r_max):
                    theta = theta_test[y, x]
                    a = x + r * cos(deg2rad(theta))
                    a_i = int(round(((a - a_min) / (a_max - a_min)) * a_len))
                    b = y + r * sin(deg2rad(theta))
                    b_i = int(round(((b - b_min) / (b_max - b_min)) * b_len))
                    k = int(round((r - r_min) / (r_max - r_min) * r_len))

                    # exclude circles with centers out of the image
                    if a < (m - 1) and b < (n - 1):
                        H[a_i, b_i, k] += 1

    return H


def find_parallels(d, theta, theta_threshold, d_threshold_min, d_threshold_max):
    peaks = column_stack((d, theta))
    print('length before filtering', len(peaks))
    add_list = []
    keep = zeros((len(peaks), len(peaks)), dtype="uint8")
    for i in range(len(peaks)):
        for j in range(len(peaks)):
            delta_d = abs(peaks[i, 0] - peaks[j, 0])
            print('the delta_d is: ', delta_d)
            delta_theta = abs(peaks[i, 1] - peaks[j, 1])
            print('the delta_theta is', delta_theta)
            if ((delta_theta < theta_threshold) & (not (i == j)) & (delta_d > d_threshold_min) & (
                    delta_d < d_threshold_max)):
                keep[i, j] = 1
    add_list = list(set((where(keep == 1))[0]))
    d_hough = peaks[add_list, 0]
    theta_hough = peaks[add_list, 1]
    return d_hough, theta_hough


def filter_circles(a,b,r,r_threshold):
    peaks = column_stack((a,b,r))
    del_list - []
    for i in range(len(peaks)):
        for j in range(len(peaks)):
            delta_r = abs(peaks[i, 2] - peaks[j, 2])
            if (delta_r < r_threshold):
                del_list.append([i])

    return d_hough, theta_hough
