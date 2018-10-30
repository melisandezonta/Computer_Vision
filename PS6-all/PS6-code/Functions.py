import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import *
import bisect


def extract_patch_template(frame, datafile, scale=None):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    file = open(datafile, 'r')
    line = file.readline()
    file.close()
    vals = line.split("  ")
    vals = [ int(float(v)) for v in vals ]
    w, h = vals[2], vals[3]
    xc, yc = int(vals[0]+floor(w/2)), int(vals[1]+floor(h/2))
    w = w if not scale else int(w * scale)
    h = h if not scale else int(h * scale)
    slicer = slice(yc - floor(h/2), yc + ceil(h/2))
    slicec = slice(xc - floor(w/2), xc + ceil(w/2))
    return frame[slicer, slicec], xc, yc


class TrackingPF():
    def __init__(self, template, initial_pose, initial_uncertainty, N):
        self.template = template
        self.initial_height = template.shape[0]
        self.initial_width  = template.shape[1]
        self.N = N

        # Initialisation of the particle cloud around the initial position
        # and size
        self.X = np.vstack([initial_pose[0,0], initial_pose[1,0], 1.0])
        self.particles = [self.clip_scale(self.X + self.draw_noise(initial_uncertainty)) for i in range(0,self.N)]

    def draw_noise(self, std):
        noise = np.zeros(std.shape)
        for i in range(std.shape[0]):
            noise[i,0] = np.random.normal(0, std[i,0], (1,1))
        return noise

    def clip_scale(self, particle):
        particle[2,0] = max(0.1, min(particle[2,0], 1.4))
        return particle

    def predict(self, precision):
        new_particles = []
        for p in self.particles:
            new_particles.append(self.clip_scale(p + self.draw_noise(precision)))
        self.particles = new_particles

    def update(self, frame, std_mse):
        W = []
        for p in self.particles: # calculate weights
            rows = self.initial_height * p[2,0]
            cols = self.initial_width * p[2,0]
            minr = max(0, min(int(p[1,0]-floor(rows/2)), frame.shape[0]-1))
            maxr = max(0, min(int(p[1,0]+ceil(rows/2)), frame.shape[0]-1))
            minc = max(0, min(int(p[0,0]-floor(cols/2)), frame.shape[1]-1))
            maxc = max(0, min(int(p[0,0]+ceil(cols/2)), frame.shape[1]-1))
            slicer = slice(minr, maxr)
            slicec = slice(minc, maxc)
            patch = cv2.cvtColor(frame[slicer, slicec], cv2.COLOR_BGR2GRAY)
            #print(int(rows), minr, maxr, int(cols), minc, maxc)
            try:
                temp = cv2.resize(self.template, (patch.shape[1], patch.shape[0]))
                MSE = np.sum(np.power(np.float64(temp) - np.float64(patch), 2))
                MSE /= patch.size
                w = np.exp(-MSE / (2 * std_mse * std_mse))
            except:
                w = 0
            W.append(w)

        W /= np.sum(W) # normalize

        Q = np.cumsum(W)

        new_particles = []
        for i in range(self.N):
            p = self.particles[bisect.bisect_left(Q, np.random.uniform(0,1))]
            new_particles.append(p)

        self.particles = new_particles

    def update_mean(self):
        X = np.zeros((3,1))
        for p in self.particles:
            X += p
        self.X = X / len(self.particles)
        return self.X

    def update_template(self, frame, alpha=None):
        rows = self.initial_height * self.X[2,0]
        cols = self.initial_width * self.X[2,0]
        minr = max(0, min(int(self.X[1,0]-floor(rows/2)), frame.shape[0]-1))
        maxr = max(0, min(int(self.X[1,0]+ceil(rows/2)), frame.shape[0]-1))
        minc = max(0, min(int(self.X[0,0]-floor(cols/2)), frame.shape[1]-1))
        maxc = max(0, min(int(self.X[0,0]+ceil(cols/2)), frame.shape[1]-1))
        slicer = slice(minr, maxr)
        slicec = slice(minc, maxc)
        patch = cv2.cvtColor(frame[slicer, slicec], cv2.COLOR_BGR2GRAY)
        temp = cv2.resize(self.template, (patch.shape[1], patch.shape[0]))
        if alpha is not None:
            temp = alpha * patch + (1 - alpha) * temp
        self.template = temp

def display_data(frame, tracker, width, height):
    result = frame.copy()
    state = tracker.X
    N = len(tracker.particles)
    distances = np.zeros((1,N))
    for i,p in enumerate(tracker.particles):
        cv2.circle(result, (int(p[0,0]), int(p[1,0])), 2, (0,255,0), -1)
        distances[0,i] = hypot(state[0]-p[0,0], state[1]-p[1,0])
    radius = int(np.sum(distances) / N)
    cv2.circle(result, (int(state[0]), int(state[1])), radius, (255,0,0), 2)
    x1, y1 = int(state[0] - floor(width/2)), int(state[1] - floor(height/2))
    x2, y2 = int(state[0] + ceil(width/2)), int(state[1] + ceil(height/2))
    cv2.rectangle(result, (x1, y1), (x2, y2), (0,0,255), 2)
    return result

def concatenate_frames(indices, frames, name):
    h, w, c = frames[0].shape
    N = len(frames)
    s = 10 # spacing
    result = np.zeros([N*h + (N-1)*s, w, c], dtype="uint8")
    result.fill(255)
    filename = name
    for i in range(N):
        result[(i * (s + h)):(i * s + (i + 1) * h), :, :] = frames[i]
        filename = "{}-{}".format(filename, indices[i])
    cv2.imwrite("{}.png".format(filename), result)
