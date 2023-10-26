import nervestitcher
import image_transform
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy
from superpoint import SuperPointData
import code


@dataclass
class ImageData:
    image: numpy.ndarray
    superpoint_data: SuperPointData


@dataclass
class RobustnessData:
    original_superpoint_data: SuperPointData
    artefacts: list[ImageData]
    embeddings: list[numpy.ndarray]
    embedding_coordinates: list[numpy.ndarray]


def calculate_distance_matrix(a1, a2):
    matrix = np.full((len(a1), len(a2)), np.Inf)
    for i in range(len(a1)):
        for j in range(len(a2)):
            matrix[i, j] = np.sqrt(((a1[i] - a2[j]) ** 2).sum())
    return matrix


scale = 0.75
movements = [
    # lambda t: (0, 0),  # NOTE: Keine Transformation
    lambda t: (0, -2 * t),  # NOTE: reine Streckung
    lambda t: (-2 * t, 0),  # NOTE: lineare Scherung
    lambda t: (-2 * t, -2 * t),  # NOTE: lineare Scherung + Streckung
    lambda t: (
        np.sin(1.8 * (np.pi * (t - 0.5))) * 0.7,
        0,
    ),  # NOTE: nicht-lineare Scherung in x-Richtung
]

images = nervestitcher.load_images_in_directory("/home/leah/Datasets/EGT7_001-A_4_snp")
images = nervestitcher.preprocess_images(images)[144:147]
# images = np.insert(images, 0, image_transform.generate_checkerboard_image(384, 384, 12), axis=0)

robustness_data = []
for image in images:
    rd = RobustnessData(None, [], [], [])
    rd.original_superpoint_data = nervestitcher.superpoint(image)
    embedding = np.zeros_like(image)
    embedding[
        rd.original_superpoint_data.coordinates[:, 1],
        rd.original_superpoint_data.coordinates[:, 0],
    ] = 1

    for movement in movements:
        artefacted_image = image_transform.apply_artefact(
            image, int(image.shape[1] * scale), int(image.shape[0] * scale), movement
        )
        superpoint_data = nervestitcher.superpoint(artefacted_image)
        rd.artefacts.append(ImageData(artefacted_image, superpoint_data))
        artefacted_embedding = image_transform.apply_artefact(
            embedding, int(embedding.shape[1] * scale), int(embedding.shape[0] * scale), movement
        )
        rd.embeddings.append(artefacted_embedding)
        coordinates = numpy.argwhere(artefacted_embedding == 1)
        # Eliminate smeared Points
        # for i in range(len(coordinates)):
        coordinates = numpy.flip(coordinates, axis=1)
        rd.embedding_coordinates.append(coordinates)
    robustness_data.append(rd)


# for rd in robustness_data:
#     for e in rd.embeddings:
#         plt.imshow(e)
#         plt.show()
code.interact(local=locals())

# # TODO: Now we need to see how many points of the retransformed coordinates are contained within the transformed data. After that we need to compare the descriptors first by vector distance, then by SuperGlue

# for image_data in superpoint_data:
#     # Original Data is used only for Descriptor comparisons
#     original_data, transformed_data, retransformed_coordinates = image_data
#     for original_sample, transformed_sample, retransformed_coordinate_sample in zip(
#         original_data, transformed_data, retransformed_coordinates
#     ):
#         c_t = (
#             transformed_sample[0][0].cpu().numpy()
#         )  # coordinates as detected by superpoint in the transformed image scan
#         c_r = retransformed_coordinate_sample  # coordinates of the original image, transformed after the fact
#         acceptable_pixel_distance = 3
#         self_distance = calculate_distance_matrix(c_t, c_t)
#         plt.imshow(self_distance < 2)
#         plt.show()
#         matrix = calculate_distance_matrix(c_t, c_r)
#         plt.imshow(matrix < acceptable_pixel_distance)
#         plt.show()
