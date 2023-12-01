import nervestitcher
import image_transform
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy
from superpoint import SuperPointData
from superglue import SuperGlueData
import code
import visualization
import pickle
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


@dataclass
class ImageData:
    image: numpy.ndarray
    superpoint_data: SuperPointData
    extra: numpy.ndarray


@dataclass
class RobustnessData:
    original: ImageData
    artefacts: list[ImageData]
    embeddings: list[ImageData]


def calculate_distance_matrix(row_values, column_values):
    matrix = np.zeros((len(row_values), len(column_values)))
    for i in range(len(row_values)):
        for j in range(len(column_values)):
            matrix[i, j] = np.sqrt(((row_values[i] - column_values[j]) ** 2).sum())
    return matrix


# def get_close_indices(coordinates, max_distance):
#     remove_list = []
#     for i in range(len(coordinates)):
#         if i in remove_list:
#             continue
#         for j in range(i + 1, len(coordinates)):
#             dist = numpy.sqrt(((coordinates[j] - coordinates[i]) ** 2).sum())
#             if dist <= max_distance:
#                 remove_list.append(j)
#     return remove_list


def deduplicate(embedding: numpy.ndarray):
    already_visited = []
    h, w = embedding.shape
    for y in range(h):
        for x in range(w):
            element = embedding[y, x]
            if element is None:
                continue
            visited = False
            for candidate in already_visited:
                if (
                    numpy.array_equal(candidate[0], element[0])
                    and numpy.array_equal(candidate[1], element[1])
                    and numpy.array_equal(candidate[2], element[2])
                ):
                    visited = True

            if visited:
                embedding[y, x] = None
            else:
                already_visited.append(element)


def create_embedding(superpoint_data: SuperPointData) -> numpy.ndarray:
    coordinates, scores, descriptors, width, height = superpoint_data
    descriptors = numpy.transpose(descriptors)
    embedding = numpy.full((height, width), None, numpy.object_)
    for c, s, d in zip(coordinates, scores, descriptors):
        embedding[c[1], c[0]] = (c, s, d)
    return embedding


def extract_data_from_embedding(embedding: numpy.ndarray) -> ImageData:
    height, width = embedding.shape
    deduplicate(embedding)
    coordinates = numpy.flip(numpy.argwhere(embedding), axis=1)

    scores = []
    descriptors = []
    original_coordinates = []
    for c in coordinates:
        c, s, d = embedding[c[1], c[0]]
        scores.append(s)
        descriptors.append(d)
        original_coordinates.append(c)
    scores = numpy.array(scores)
    descriptors = numpy.array(descriptors)
    descriptors = numpy.transpose(descriptors)
    original_coordinates = numpy.array(original_coordinates)

    superpoint_data = SuperPointData(coordinates, scores, descriptors, width, height)
    image = numpy.zeros_like(embedding, numpy.float32)
    image[coordinates[:, 1], coordinates[:, 0]] = 1
    return ImageData(image, superpoint_data, original_coordinates)


def extract_robustness_data(images, scale=0.75, movements=[]):
    robustness_data = []
    for image in images:
        rd = RobustnessData(None, [], [])
        rd.original = ImageData(image, nervestitcher.superpoint(image), None)

        embedding = create_embedding(rd.original.superpoint_data)

        for movement in movements:
            artefacted_image = image_transform.apply_artefact(
                image, int(image.shape[1] * scale), int(image.shape[0] * scale), movement
            )
            superpoint_data = nervestitcher.superpoint(artefacted_image)
            rd.artefacts.append(ImageData(artefacted_image, superpoint_data, None))

            artefacted_embedding = image_transform.apply_artefact(
                embedding,
                int(embedding.shape[1] * scale),
                int(embedding.shape[0] * scale),
                movement,
                numpy.object_,
            )
            embedding_data = extract_data_from_embedding(artefacted_embedding)
            # img = numpy.zeros_like(artefacted_embedding, numpy.float32)
            # for y in range(artefacted_embedding.shape[0]):
            #     for x in range(artefacted_embedding.shape[1]):
            #         img[y, x] = 0 if artefacted_embedding[y, x] is None else 1
            # plt.imshow(img, cmap="Greys_r")
            # plt.show()
            # plt.imshow(embedding_data.image, cmap="Greys_r")
            # plt.tight_layout()
            # plt.show()

            rd.embeddings.append(embedding_data)
        robustness_data.append(rd)
    return robustness_data


def viz_robustness_data(d: RobustnessData):
    rows = 1
    cols = 1  # len(d.artefacts) + 1
    plt.figure(
        figsize=(d.original.superpoint_data.width / 100, d.original.superpoint_data.height / 100),
        dpi=100,
    )
    plt.imshow(d.original.image, cmap="Greys_r")
    plt.scatter(
        d.original.superpoint_data.coordinates[:, 0],
        d.original.superpoint_data.coordinates[:, 1],
        marker="x",
        color="orange",
        s=20,
    )
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.axis("off")
    plt.show()
    for a in d.artefacts:
        plt.figure(
            figsize=(
                a.superpoint_data.width / 100,
                a.superpoint_data.height / 100,
            ),
            dpi=100,
        )
        plt.imshow(a.image, cmap="Greys_r")
        plt.scatter(
            a.superpoint_data.coordinates[:, 0],
            a.superpoint_data.coordinates[:, 1],
            marker="x",
            color="magenta",
            s=20,
        )
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.axis("off")
        plt.show()

    # fig = plt.figure(figsize=(5, 5))
    # ax1 = fig.add_subplot(rows, cols, 1)
    # ax1.imshow(d.artefacts[3].image, cmap="Greys_r")
    # h, w = d.original.image.shape
    # ax1.axhline(h * 0.125, xmin=0.125, xmax=0.875, color="blue", lw=2)
    # ax1.axhline(h * 0.875, xmin=0.125, xmax=0.875, color="blue", lw=2)
    # ax1.axvline(w * 0.125, ymin=0.125, ymax=0.875, color="blue", lw=2)
    # ax1.axvline(w * 0.875, ymin=0.125, ymax=0.875, color="blue", lw=2)
    # ax1.scatter(
    #     d.original.superpoint_data.coordinates[:, 0],
    #     d.original.superpoint_data.coordinates[:, 1],
    #     marker="x",
    #     color="orange",
    #     s=15,
    # )
    # # for i, (a, e) in enumerate(zip(d.artefacts, d.embeddings)):
    #     ax_a = fig.add_subplot(rows, cols, i + 2)
    #     ax_a.imshow(a.image, cmap="Greys_r")
    #     ax_a.scatter(
    #         a.superpoint_data.coordinates[:, 0],
    #         a.superpoint_data.coordinates[:, 1],
    #         marker="x",
    #         color="magenta",
    #         s=15,
    #     )
    #     ax_a.set_xticks([])
    #     ax_a.set_yticks([])
    #     ax_e = fig.add_subplot(rows, cols, i + 2 + cols)
    #     ax_e.imshow(e.image, cmap="Greys")
    #     ax_e.scatter(
    #         e.superpoint_data.coordinates[:, 0],
    #         e.superpoint_data.coordinates[:, 1],
    #         marker="x",
    #         color="lightblue",
    #         s=15,
    #     )
    #     ax_e.set_xticks([])
    #     ax_e.set_yticks([])
    # fig.tight_layout()


# def get_coordinates_and_descriptors_from_superglue_data(
#     superglue_data: SuperGlueData,
#     superpoint_data_a: SuperPointData,
#     superpoint_data_b: SuperPointData,
# ):
#     valid = superglue_data.indices_a > -1
#     coordinates_a = superpoint_data_a.coordinates[valid]
#     descriptors_a = superpoint_data_a.descriptors[:, valid]
#     coordinates_b = superpoint_data_b.coordinates[superglue_data.indices_a[valid]]
#     descriptors_b = superpoint_data_b.descriptors[:, superglue_data.indices_a[valid]]
#     return coordinates_a, coordinates_b, descriptors_a, descriptors_b


movements = [
    # lambda t: (0, 0),  # NOTE: Keine Transformation
    lambda t: (0, -3 * t),  # NOTE: reine Streckung
    lambda t: (-2 * t, 0),  # NOTE: lineare Scherung
    lambda t: (-2 * t, -3 * t),  # NOTE: lineare Scherung + Streckung
    lambda t: (
        np.sin(1.8 * (np.pi * (t - 0.5))) * 0.7,
        0,
    ),  # NOTE: nicht-lineare Scherung in x-Richtung
]

images = nervestitcher.load_images_in_directory(
    "/mnt/home/leah/Datasets/Preprocessed/robustness_testing_set/"
)[20:21]
# images = np.insert(images, 0, image_transform.generate_checkerboard_image(384, 384, 12), axis=0)

robustness_data = extract_robustness_data(images, scale=0.75, movements=movements)

for d in robustness_data:
    viz_robustness_data(d)


exit(0)

statistics = []
max_distance = 3
for rd in robustness_data:
    stats = {
        "unique_count_artefact": [],
        "unique_count_original": [],
        "non_unique_count": [],
        "artefact_count": [],
        "original_count": [],
        "match_count": [],
        "superglue_subset_count": [],
        "superglue_subset_valid_count": [],
    }
    for a, e in zip(rd.artefacts, rd.embeddings):
        distances = calculate_distance_matrix(
            a.superpoint_data.coordinates, e.superpoint_data.coordinates
        )
        matches = distances <= max_distance

        # plt.imshow(matches, cmap="gnuplot")
        # # plt.imshow(distances, cmap="gnuplot")
        # plt.colorbar()
        # plt.imshow(e.image, cmap="Greys_r")
        # plt.tight_layout()
        # plt.show()

        only_in_artefact = numpy.argwhere(numpy.count_nonzero(matches, axis=1) == 0)
        only_in_original = numpy.argwhere(numpy.count_nonzero(matches, axis=0) == 0)
        unique_count_artefact = len(only_in_artefact)
        unique_count_original = len(only_in_original)
        non_unique_count = matches.shape[0] - unique_count_artefact
        stats["unique_count_artefact"].append(unique_count_artefact)
        stats["unique_count_original"].append(unique_count_original)
        stats["non_unique_count"].append(non_unique_count)
        stats["artefact_count"].append(matches.shape[0])
        stats["original_count"].append(matches.shape[1])

        h, w = matches.shape
        coordinates_a = []
        scores_a = []
        descriptors_a = []
        coordinates_e = []
        scores_e = []
        descriptors_e = []
        for row in range(h):
            for col in range(w):
                if matches[row, col] == 1:
                    coordinates_a.append(a.superpoint_data.coordinates[row])
                    scores_a.append(a.superpoint_data.scores[row])
                    descriptors_a.append(a.superpoint_data.descriptors[:, row])
                    coordinates_e.append(e.extra[col])
                    scores_e.append(e.superpoint_data.scores[col])
                    descriptors_e.append(e.superpoint_data.descriptors[:, col])
        coordinates_a = numpy.array(coordinates_a)
        scores_a = numpy.array(scores_a)
        descriptors_a = numpy.array(descriptors_a).transpose()
        coordinates_e = numpy.array(coordinates_e)
        scores_e = numpy.array(scores_e)
        descriptors_e = numpy.array(descriptors_e).transpose()

        superpoint_data_a = SuperPointData(
            coordinates_a,
            scores_a,
            descriptors_a,
            a.superpoint_data.width,
            a.superpoint_data.height,
        )
        superpoint_data_e = SuperPointData(
            coordinates_e,
            scores_e,
            descriptors_e,
            e.superpoint_data.width,
            e.superpoint_data.height,
        )

        superglue_data: SuperGlueData = nervestitcher.superglue(
            superpoint_data_a, superpoint_data_e
        )
        valid = superglue_data.indices_a > -1
        sg_coordinates_a = superpoint_data_a.coordinates[valid]
        sg_coordinates_e = superpoint_data_e.coordinates[superglue_data.indices_a[valid]]

        # finally of this subset find the ones that have been matched the same way
        same_match_count = 0
        for c_a, c_e in zip(sg_coordinates_a, sg_coordinates_e):
            index = np.where(coordinates_a == c_a)
            if numpy.array_equal(coordinates_e[index], c_e):
                same_match_count += 1
        stats["match_count"].append(len(coordinates_a))
        stats["superglue_subset_count"].append(len(sg_coordinates_a))
        stats["superglue_subset_valid_count"].append(same_match_count)
    statistics.append(stats)

split_dict = []
for d in statistics:
    new_dict = {}
    for k, v in d.items():
        for i, e in enumerate(v):
            new_dict[f"{k}_{i}"] = e
    split_dict.append(new_dict)

df = pd.DataFrame(split_dict)
# df.to_csv("./data/robustnes_statistics.csv")
