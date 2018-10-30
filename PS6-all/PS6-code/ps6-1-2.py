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
predict_std = np.vstack([5.0, 5.0, 0.0])
N = 100
scale_small = 0.1
scale_large = 2.5

print("""Parameters:
MSE standard deviation = {}
Initial noise standard deviation = {}
Predict noise standard deviation = {}
Number of particles = {}
Small scale = {}
Large scale = {}
""".format(std_mse, initial_std.T, predict_std.T, N, scale_small, scale_large))

indices = [28, 84, 144]

video = cv2.VideoCapture()

video.open("{}.avi".format(name))

# Get first frame
_, first_frame = video.read()

# Extract small template
temp_small, xs, ys = extract_patch_template(first_frame, "{}.txt".format(name), scale_small)
hs, ws = temp_small.shape
cv2.imwrite("ps6-1-2-template-small.png", temp_small)

# Extract large template
temp_large, xl, yl = extract_patch_template(first_frame, "{}.txt".format(name), scale_large)
hl, wl = temp_large.shape
cv2.imwrite("ps6-1-2-template-large.png", temp_large)

# Create trackers
tracker_small = TrackingPF(temp_small, np.vstack([xs, ys]), initial_std, N)
tracker_large = TrackingPF(temp_large, np.vstack([xl, yl]), initial_std, N)

images_small = []
images_large = []

i = 1
while True:
    read, frame = video.read()
    if not read or i == 150:
        break

    # Small template tracker
    tracker_small.predict(predict_std)
    tracker_small.update(frame, std_mse)
    state_small = tracker_small.update_mean()
    frame_small = display_data(frame, tracker_small, ws, hs)

    # Large template tracker
    tracker_large.predict(predict_std)
    tracker_large.update(frame, std_mse)
    state_large = tracker_large.update_mean()
    frame_large = display_data(frame, tracker_large, wl, hl)

    if i in indices:
        images_small.append(frame_small)
        images_large.append(frame_large)
    i += 1

concatenate_frames(indices, images_small, "ps6-1-2-frames-small")
concatenate_frames(indices, images_large, "ps6-1-2-frames-large")

video.release()

plt.show()
