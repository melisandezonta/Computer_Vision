import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS6-all/PS6-input/")


# Parameters
name = "pedestrians"
std_mse = 10.0
initial_std = np.vstack([0.1, 0.1, 0.1])
predict_std = np.vstack([0.5, 0.5, 0.2])
N = 200

print("""Parameters:
MSE standard deviation = {}
Initial noise standard deviation = {}
Predict noise standard deviation = {}
Number of particles = {}
""".format(std_mse, initial_std.T, predict_std.T, N))

indices = [40, 100, 240]

video = cv2.VideoCapture()

video.open("{}.avi".format(name))

# Get first frame and extract template
_, first_frame = video.read()
template, xc, yc = extract_patch_template(first_frame, "{}.txt".format(name))
height, width = template.shape
cv2.imwrite("ps6-3-1-template.png", template)

tracker = TrackingPF(template, np.vstack([xc, yc]), initial_std, N)

images = []

i = 1
while True:
    read, frame = video.read()
    if not read or i == 241:
        break
    tracker.predict(predict_std)
    tracker.update(frame, std_mse)
    state = tracker.update_mean()
    tracker.update_template(frame)
    h, w = tracker.template.shape
    frame = display_data(frame, tracker, w, h)
    if i in indices:
        images.append(frame)
    i += 1

concatenate_frames(indices, images, "ps6-3-1-frames")

video.release()

plt.show()
