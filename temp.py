import cv2
import numpy as np
import nervestitcher as ns
from superpoint import SuperPointData
import visualization
from matplotlib import pyplot as plt

images = ns.load_images_in_directory("/home/leah/Datasets/EGT7_001-A_4_snp")
prep = ns.preprocess_images(images)
# prep *= 255

indices = [303, 301, 266, 257]
imgs = [prep[i] for i in indices]
superpoint_data: list[tuple[np.ndarray, SuperPointData]] = [
    (img, ns.superpoint(img)) for img in imgs
]
for spd in superpoint_data:
    fig, ax = visualization.image_grid_with_coordinates(
        [spd[0]], [spd[1].coordinates], 1, 1, size=30, colors=[spd[1].scores], marker="x"
    )
    fig.show()
    plt.show()


# for i in indices:
#     cv2.imwrite(f"EGT7_001-A_4_{i}.png", prep[i].astype(np.uint8))
#     # cv2.imshow(f"{i}", (prep[i]).astype(np.uint8))

# # while cv2.waitKey(0) != 27:
# #     pass

# # cv2.destroyAllWindows()
