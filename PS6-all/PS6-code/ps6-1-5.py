import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from Functions import *

os.chdir("/Users/melisandezonta/Documents/Documents/Documents/GTL_courses_second_semester/Computer-Vision/PS6-all/PS6-input/")


# Parameters
name = "noisy_debate"
std_mse_low  = 5.0
std_mse_mid  = 10.0
std_mse_high = 15.0
initial_std = np.vstack([5.0, 5.0, 0.0])
predict_std = np.vstack([5.0, 5.0, 0.0])
N = 30

print("""Parameters:
MSE standard deviation (low) = {}
MSE standard deviation (mid) = {}
MSE standard deviation (high) = {}
Initial noise standard deviation = {}
Predict noise standard deviation = {}
Number of particles = {}
""".format(std_mse_low, std_mse_mid, std_mse_high, initial_std.T, predict_std.T, N))

indices = [14, 32, 46]

video = cv2.VideoCapture()

video.open("{}.avi".format(name))

# Get first frame and extract template
_, first_frame = video.read()
template, xc, yc = extract_patch_template(first_frame, "{}.txt".format(name))
height, width = template.shape
cv2.imwrite("ps6-1-5-template.png", template)

tracker1 = TrackingPF(template, np.vstack([xc, yc]), initial_std, N)
tracker2 = TrackingPF(template, np.vstack([xc, yc]), initial_std, N)
tracker3 = TrackingPF(template, np.vstack([xc, yc]), initial_std, N)

images1 = []
images2 = []
images3 = []

i = 1
while True:
    read, frame = video.read()
    if not read or i == 150:
        break

    tracker1.predict(predict_std)
    tracker1.update(frame, std_mse_low)
    state1 = tracker1.update_mean()
    frame1 = display_data(frame, tracker1, width, height)

    tracker2.predict(predict_std)
    tracker2.update(frame, std_mse_mid)
    state2 = tracker2.update_mean()
    frame2 = display_data(frame, tracker2, width, height)

    tracker3.predict(predict_std)
    tracker3.update(frame, std_mse_high)
    state3 = tracker3.update_mean()
    frame3 = display_data(frame, tracker3, width, height)

    if i in indices:
        images1.append(frame1)
        images2.append(frame2)
        images3.append(frame3)
    i += 1

concatenate_frames(indices, images1, "ps6-1-5-frames1")
concatenate_frames(indices, images2, "ps6-1-5-frames2")
concatenate_frames(indices, images3, "ps6-1-5-frames3")

video.release()

plt.show()
