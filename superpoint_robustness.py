import nervestitcher
import image_transform
import numpy as np
from matplotlib import pyplot as plt
import visualization
import torch
import code


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
images = np.insert(images, 0, image_transform.generate_checkerboard_image(384, 384, 12), axis=0)

image_collections = [
    [
        image_transform.apply_artefact(
            image, int(image.shape[1] * scale), int(image.shape[0] * scale), movement
        )
        for movement in movements
    ]
    for image in images
]

transformed_superpoint_data = [
    [
        nervestitcher.superpoint(torch.from_numpy(image).float()[None][None])
        for image in image_collection
    ]
    for image_collection in image_collections
]

original_superpoint_data = []
coordinate_embeddings = []
for image in images:
    data = nervestitcher.superpoint(torch.from_numpy(image).float()[None][None])
    coordinates, _, _ = data
    embedding = np.zeros_like(image)
    coordinates = coordinates[0].cpu().numpy()
    embedding[coordinates[:, 1], coordinates[:, 0]] = 1
    coordinate_embeddings.append(embedding)
    original_superpoint_data.append(data)

transformed_embeddings = [
    [
        np.flip(
            np.argwhere(
                image_transform.apply_artefact(
                    embedding,
                    int(embedding.shape[1] * scale),
                    int(embedding.shape[0] * scale),
                    movement,
                )
            ),
            axis=1,
        )
        for movement in movements
    ]
    for embedding in coordinate_embeddings
]

superpoint_data = [
    [original_data, transformed_data, retransformed_coordinates]
    for (original_data, transformed_data, retransformed_coordinates) in zip(
        original_superpoint_data, transformed_superpoint_data, transformed_embeddings
    )
]

# TODO: Now we need to see how many points of the retransformed coordinates are contained within the transformed data. After that we need to compare the descriptors first by vector distance, then by SuperGlue

for image_data in superpoint_data:
    # Original Data is used only for Descriptor comparisons
    original_data, transformed_data, retransformed_coordinates = image_data
    for original_sample, transformed_sample, retransformed_coordinate_sample in zip(
        original_data, transformed_data, retransformed_coordinates
    ):
        c_t = (
            transformed_sample[0][0].cpu().numpy()
        )  # coordinates as detected by superpoint in the transformed image scan
        c_r = retransformed_coordinate_sample  # coordinates of the original image, transformed after the fact
        acceptable_pixel_distance = 3
        self_distance = calculate_distance_matrix(c_t, c_t)
        plt.imshow(self_distance < 2)
        plt.show()
        matrix = calculate_distance_matrix(c_t, c_r)
        plt.imshow(matrix < acceptable_pixel_distance)
        plt.show()
