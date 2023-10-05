import numpy
import nervestitcher
import fusion
import image_transform
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import cv2
import torch
import config

images = nervestitcher.load_images_in_directory("/home/leah/Datasets/EGT7_001-A_4_snp")
images = nervestitcher.preprocess_images(images)

_, scores, _ = fusion.load_interest_point_data("./data/EGT7_001-A_4_snp_IP_0.005.pkl")
counts = [len(score) for score in scores]
above_average_scores = numpy.array([numpy.count_nonzero(s > numpy.average(s)) for s in scores])
value = above_average_scores / counts
plt.bar(range(len(above_average_scores)), value)
plt.show()

for image in images[value < 0.25]:
    plt.imshow(image, cmap="Greys_r")
    plt.show()
# averages = [numpy.average(score) for score in scores]
# mean = numpy.mean(averages)
# stddev = numpy.std(averages)
# z = (averages - mean) / stddev

# plt.bar(range(len(z)), z / counts)
# plt.show()
