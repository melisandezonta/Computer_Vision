import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS6-all/PS6-input/")

# Parameters
name = "pres_debate"
std_mse = 10.0
initial_std = np.vstack([5.0, 5.0, 0.0])
predict_std = np.vstack([20.0, 20.0, 0.0])
N = 100
alpha = 0.2

print("""Parameters:
MSE standard deviation = {}
Initial noise standard deviation = {}
Predict noise standard deviation = {}
Number of particles = {}
Alpha = {}
""".format(std_mse, initial_std.T, predict_std.T, N, alpha))

indices = [15, 50, 140]

video = cv2.VideoCapture()

video.open("{}.avi".format(name))

# Get first frame and extract template
_, first_frame = video.read()
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
xc, yc = 575, 445
height, width = 50, 50
slicer = slice(yc - floor(height/2), yc + ceil(height/2))
slicec = slice(xc - floor(width/2), xc + ceil(width/2))
template = first_frame[slicer, slicec]
cv2.imwrite("ps6-2-1-template.png", template)

tracker = TrackingPF(template, np.vstack([xc, yc]), initial_std, N)

images = []

i = 1
while True:
    read, frame = video.read()
    if not read or i == 150:
        break
    tracker.predict(predict_std)
    tracker.update(frame, std_mse)
    state = tracker.update_mean()
    tracker.update_template(frame, alpha)
    frame = display_data(frame, tracker, width, height)
    if i in indices:
        images.append(frame)
    i += 1

concatenate_frames(indices, images, "ps6-2-1-frames")

video.release()

plt.show()
