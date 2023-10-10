import numpy
import nervestitcher
import fusion
import image_transform
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import torch
import config
import code
from functools import reduce
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# images = nervestitcher.load_images_in_directory("/home/leah/Datasets/EGT7_001-A_3_snp")
# images = nervestitcher.preprocess_images(images)
# _, scores, _ = fusion.load_interest_point_data("./data/EGT7_001-A_4_snp_IP_0.005.pkl")
# counts = [len(score) for score in scores]
# above_average_scores = numpy.array([numpy.count_nonzero(s > numpy.average(s)) for s in scores])
# value = above_average_scores / counts
# plt.bar(range(len(above_average_scores)), value)
# plt.show()

# for image in images[value < 0.25]:
#     plt.imshow(image, cmap="Greys_r")
#     plt.show()
# # averages = [numpy.average(score) for score in scores]
# # mean = numpy.mean(averages)
# # stddev = numpy.std(averages)
# # z = (averages - mean) / stddev

# # plt.bar(range(len(z)), z / counts)
# # plt.show()

# coordinates, scores, descriptors = fusion.load_interest_point_data(
#     "./data/EGT7_001-A_3_snp_IP_0.005.pkl"
# )
# adjacency = fusion.generate_adjacency_matrix_reduced(images, coordinates, scores, descriptors)
# fusion.save_adjacency_matrix("./data/EGT7_001-A_3_snp_ADJ_REDUCED.pkl", adjacency)

# adjacency = fusion.load_adjacency_matrix("./data/EGT7_001-A_3_snp_ADJ_REDUCED.pkl")
# artefact_indices = [
#     16,
#     41,
#     100,
#     141,
#     177,
#     189,
#     232,
#     244,
#     351,
#     372,
#     504,
#     540,
#     544,
#     566,
#     680,
#     772,
#     783,
#     921,
#     987,
#     997,
#     1015,
#     1117,
#     1136,
#     1227,
#     1230,
#     1234,
#     1397,
#     1403,
# ]


# def z_score_test(adjacency, kernel_size=51, threshold=-1.5):
#     n = 3
#     transforms_g = []
#     match_count_pre_g = []
#     match_count_post_g = []
#     scores_a_g = []
#     scores_b_g = []
#     for i in range(len(adjacency) - n):
#         transforms, match_count_pres, match_count_posts, scores_as, scores_bs = [], [], [], [], []
#         for j in range(1, n + 1):
#             transform, match_count_pre, match_count_post, scores_a, scores_b = adjacency[i][i + j]
#             transforms.append(transform)
#             match_count_pres.append(match_count_pre)
#             match_count_posts.append(match_count_post)
#             scores_as.append(scores_a)
#             scores_bs.append(scores_b)
#         transforms_g.append(transforms)
#         match_count_pre_g.append(match_count_pres)
#         match_count_post_g.append(match_count_posts)
#         scores_a_g.append(scores_as)
#         scores_b_g.append(scores_bs)

#     match_count_pre_g = numpy.array(match_count_pre_g)

#     indices_g = []
#     for i in range(len(match_count_pre_g[0])):
#         x = match_count_pre_g[:, i]
#         x_ = numpy.pad(x, kernel_size // 2, mode="symmetric")
#         x_local_mean = sliding_window_view(x_, kernel_size).mean(axis=1)
#         x_local_std = sliding_window_view(x_, kernel_size).std(axis=1)
#         z = (x - x_local_mean) / x_local_std
#         indices = numpy.nonzero(z < threshold)[0]
#         indices_g.append(indices)

#     a = numpy.intersect1d(indices_g[0], indices_g[1])
#     b = numpy.intersect1d(indices_g[0], indices_g[2])
#     c = numpy.intersect1d(indices_g[1], indices_g[2])
#     common = reduce(numpy.union1d, [a, b, c])
#     return common


# def svm_scoring(adjacency):
#     image_count = len(adjacency)
#     samples = []
#     for i in range(image_count - 1):
#         _, match_count_pre, match_count_post, scores_a, scores_b = adjacency[i][i + 1]
#         delta = match_count_pre - match_count_post
#         mean_score_a = scores_a.mean()
#         median_score_a = numpy.median(scores_a)
#         std_score_a = scores_a.std()
#         mean_score_b = scores_b.mean()
#         median_score_b = numpy.median(scores_b)
#         std_score_b = scores_b.std()
#         samples.append(
#             [
#                 mean_score_a,
#                 mean_score_b,
#                 median_score_a,
#                 median_score_b,
#                 std_score_a,
#                 std_score_b,
#                 match_count_pre,
#                 match_count_post,
#             ]
#         )
#     samples = numpy.array(samples)
#     labels = [1 if i in artefact_indices else 0 for i in range(image_count - 1)]

#     # X_train, X_test, Y_train, Y_test = train_test_split(samples, labels, test_size=0.1)
#     clf = make_pipeline(StandardScaler(), SVC(gamma="scale", kernel="poly", degree=10))
#     clf.fit(samples, labels)
#     predictions = clf.predict(samples)
#     for i, p in enumerate(predictions):
#         if p == 1:
#             print(i)
#             plt.imshow(images[i], cmap="Greys_r")
#             plt.show()


# # svm_scoring(adjacency)


# plt.scatter(adjacency[10, 11][5][:, 0], adjacency[10, 11][5][:, 1], s=3, c="green")
# # plt.scatter(adjacency[10, 11][6][:, 0], adjacency[10, 11][6][:, 1], s=3, c="red")
# plt.scatter(adjacency[10, 12][5][:, 0], adjacency[10, 12][5][:, 1], s=3, c="blue")
# plt.scatter(adjacency[10, 13][5][:, 0], adjacency[10, 13][5][:, 1], s=3, c="cyan")
# # plt.scatter(adjacency[10, 15][6][:, 0], adjacency[10, 15][6][:, 1], s=3, c="blue")

# plt.show()
