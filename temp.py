import cv2
import numpy as np

# import nervestitcher as ns
# from superpoint import SuperPointData
import visualization
from matplotlib import pyplot as plt

# import fusion
import pickle
import pandas as pd
import seaborn as sns

# m = fusion.load_raw_match_matrix("./data/matchdata/EGT7_001-A_4.pkl")
# m = [m[i, i + 1] for i in range(len(m) - 1)]


# def per_image_mask_to_image_pair_mask(mask):
#     pair_mask = [False for _ in range(len(mask) - 1)]
#     for i, e in enumerate(mask):
#         if e == True:
#             if i > 0:
#                 pair_mask[i - 1] = True
#             if i < len(pair_mask):
#                 pair_mask[i] = True
#     return pair_mask


# f = open(f"./data/labels/EGT7_001_A_4_snp.pkl", "rb")
# mask = pickle.load(f)
# f.close()
# artefact_mask = [True if e == 1 else False for e in mask]
# pair_mask = per_image_mask_to_image_pair_mask(artefact_mask)
# for i, pair in enumerate(m):
#     if pair_mask[i] == True:
#         plt.imshow(np.exp(pair.superglue_data.scores[:-1, :-1]), cmap="gnuplot")
#         plt.colorbar()
#         plt.tight_layout()
#         plt.show()
# images = ns.load_images_in_directory("/home/leah/Datasets/EGT7_001-A_4_snp")
# prep = ns.preprocess_images(images)
# # prep *= 255

# indices = [303, 301, 266, 257]
# imgs = [prep[i] for i in indices]
# superpoint_data: list[tuple[np.ndarray, SuperPointData]] = [
#     (img, ns.superpoint(img)) for img in imgs
# ]
# for spd in superpoint_data:
#     fig, ax = visualization.image_grid_with_coordinates(
#         [spd[0]], [spd[1].coordinates], 1, 1, size=30, colors=[spd[1].scores], marker="x"
#     )
#     fig.show()
#     plt.show()


# # for i in indices:
# #     cv2.imwrite(f"EGT7_001-A_4_{i}.png", prep[i].astype(np.uint8))
# #     # cv2.imshow(f"{i}", (prep[i]).astype(np.uint8))

# # # while cv2.waitKey(0) != 27:
# # #     pass

# # # cv2.destroyAllWindows(

# ipd = fusion.load_interest_point_data("data/ipdata/EGT7_001-A_4.pkl")
# sgd = ns.superglue(ipd[10], ipd[10])
# plt.imshow(np.exp(sgd.scores[:-1, :-1]), cmap="gnuplot")
# plt.colorbar()
# plt.tight_layout()
# plt.show()

# image = cv2.imread(
#     "/home/leah/Datasets/EGT7_001-A_4_snp/EGT7_001-A_4_00050.tif", cv2.IMREAD_GRAYSCALE
# )
# a = image[:-84, :-84]
# b = image[:-84, 84:]

# sp_a = ns.superpoint(a)
# sp_b = ns.superpoint(b)
# print(sp_a)
# print(sp_b)

# sgd = ns.superglue(sp_a, sp_b)

# plt.imshow(np.exp(sgd.scores[:-1, :-1]), cmap="gnuplot")
# plt.colorbar()
# plt.tight_layout()
# plt.show()

uni = np.random.uniform(-20, 0, 10000)
sns.histplot(uni)
plt.show()
uni_exp = np.exp(uni)
sns.histplot(uni_exp)
# plt.hist(uni, 100)
plt.show()
# sns.set_palette("colorblind")
# sns.set_context("paper")
# sns.set_style("ticks")
# # ip = pd.read_csv("./data/ipdata_full.csv")
# # ip2 = ip.loc[ip["dataset"] == "EGT7_001_A_3"].reset_index()
# df = pd.read_csv("./data/matchdata_full.csv")
# df2 = df.loc[df["dataset"] == "EGT7_001_A_3"].reset_index()
# ax = sns.lineplot(df2, x=df2.index, y="valid_count")
# # ax = sns.lineplot(df2, x=df2.index, y="mean_score")
# ax.set_xlabel("Index")
# # ax.set_ylabel("Mean Score")
# ax.set_ylabel("Valid Count")
# plt.tight_layout()
# plt.show()
# # print(df)
