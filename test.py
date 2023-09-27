from image_transform import (
    apply_artefact,
    apply_gaussian_noise,
    generate_checkerboard_image,
)
import matplotlib.pyplot as plt
import numpy
import cv2
import nervestitcher as ns
import torch
import fusion


# images = ns.load_images_in_directory("/home/leah/Dataset/EGT7_001-A_4_snp")
# images = ns.preprocess_images(numpy.array(images))

# coordinates, scores, descriptors = fusion.generate_interest_point_data(images, 0.005)
# fusion.save_interest_point_data("./EGT7_001-A_4_snp_IP_0.005.pkl", coordinates, scores, descriptors)
_, scores, _ = fusion.load_interest_point_data("./EGT7_001-A_4_snp_IP_0.005.pkl")
scores = [score.cpu().numpy() for score in scores]
csv_string = ""
for s in scores:
    s = [str(e) for e in s]
    csv_string += ",".join(s)
    csv_string += "\n"


with open("scores.csv", "w+") as csvfile:
    csvfile.write(csv_string)
# plt.show()
# matching_coordinates, index_pairs = fusion.generate_matching_data_temporal(
#     images, coordinates, scores, descriptors
# )
# transforms = fusion.generate_local_transforms_from_coordinate_pair_list(matching_coordinates)
# translations = [numpy.array([m[0][2], m[1][2]]) for m in transforms]
