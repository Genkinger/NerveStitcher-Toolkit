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
    matrix = np.full((len(row_values), len(column_values)), np.Inf)
    for i in range(len(row_values)):
        for j in range(len(column_values)):
            matrix[i, j] = np.sqrt(((row_values[i] - column_values[j]) ** 2).sum())
    return matrix


def get_close_indices(coordinates, max_distance):
    remove_list = []
    for i in range(len(coordinates)):
        if i in remove_list:
            continue
        for j in range(i + 1, len(coordinates)):
            dist = numpy.sqrt(((coordinates[j] - coordinates[i]) ** 2).sum())
            if dist <= max_distance:
                remove_list.append(j)
    return remove_list


def create_embedding(superpoint_data: SuperPointData) -> numpy.ndarray:
    coordinates, scores, descriptors, width, height = superpoint_data
    descriptors = numpy.transpose(descriptors)
    embedding = numpy.full((height, width), None, numpy.object_)
    for c, s, d in zip(coordinates, scores, descriptors):
        embedding[c[1], c[0]] = (s, d, c)
    return embedding


def extract_data_from_embedding(embedding: numpy.ndarray) -> ImageData:
    height, width = embedding.shape
    coordinates = numpy.argwhere(embedding)

    delete_list = get_close_indices(coordinates, 3)
    for index in delete_list:
        embedding[coordinates[index, 0], coordinates[index, 1]] = None
    coordinates = numpy.flip(
        numpy.array([c for i, c in enumerate(coordinates) if i not in delete_list]), axis=1
    )

    scores = []
    descriptors = []
    original_coordinates = []
    for c in coordinates:
        s, d, c = embedding[c[1], c[0]]
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
            rd.embeddings.append(embedding_data)
        robustness_data.append(rd)
    return robustness_data


def viz_robustness_data(d: RobustnessData):
    rows = 2
    cols = len(d.artefacts) + 1
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(d.original.image, cmap="Greys_r")
    # h, w = d.original.image.shape
    # ax1.axhline(h * 0.125, xmin=0.125, xmax=0.875, color="blue", lw=2)
    # ax1.axhline(h * 0.875, xmin=0.125, xmax=0.875, color="blue", lw=2)
    # ax1.axvline(w * 0.125, ymin=0.125, ymax=0.875, color="blue", lw=2)
    # ax1.axvline(w * 0.875, ymin=0.125, ymax=0.875, color="blue", lw=2)
    ax1.scatter(
        d.original.superpoint_data.coordinates[:, 0],
        d.original.superpoint_data.coordinates[:, 1],
        marker="x",
        color="orange",
        s=15,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    for i, (a, e) in enumerate(zip(d.artefacts, d.embeddings)):
        ax_a = fig.add_subplot(rows, cols, i + 2)
        ax_a.imshow(a.image, cmap="Greys_r")
        ax_a.scatter(
            a.superpoint_data.coordinates[:, 0],
            a.superpoint_data.coordinates[:, 1],
            marker="x",
            color="magenta",
            s=15,
        )
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        ax_e = fig.add_subplot(rows, cols, i + 2 + cols)
        ax_e.imshow(e.image, cmap="Greys")
        ax_e.scatter(
            e.superpoint_data.coordinates[:, 0],
            e.superpoint_data.coordinates[:, 1],
            marker="x",
            color="lightblue",
            s=15,
        )
        ax_e.set_xticks([])
        ax_e.set_yticks([])
    fig.tight_layout()
    return fig


def get_coordinates_and_descriptors_from_superglue_data(
    superglue_data: SuperGlueData,
    superpoint_data_a: SuperPointData,
    superpoint_data_b: SuperPointData,
):
    valid = superglue_data.indices_a > -1
    coordinates_a = superpoint_data_a.coordinates[valid]
    descriptors_a = superpoint_data_a.descriptors[:, valid]
    coordinates_b = superpoint_data_b.coordinates[superglue_data.indices_a[valid]]
    descriptors_b = superpoint_data_b.descriptors[:, superglue_data.indices_a[valid]]
    return coordinates_a, coordinates_b, descriptors_a, descriptors_b


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
    "/home/leah/Datasets/Preprocessed/robustness_testing_set/"
)[:10]
images = np.insert(images, 0, image_transform.generate_checkerboard_image(384, 384, 12), axis=0)

robustness_data = extract_robustness_data(images, scale=0.75, movements=movements)

statistics = []
max_distance = 3
for rd in robustness_data:
    stats = {
        "unique_count_artefact": [],
        "unique_count_original": [],
        "non_unique_count": [],
        "artefact_count": [],
        "original_count": [],
    }
    for a, e in zip(rd.artefacts, rd.embeddings):
        distances = calculate_distance_matrix(
            a.superpoint_data.coordinates, e.superpoint_data.coordinates
        )
        print("MEEP")
        matches = distances <= max_distance
        # plt.imshow(matches)
        # plt.plot()
        # plt.imshow(distances)
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
        statistics.append(stats)
        superglue_data = nervestitcher.superglue(a.superpoint_data, rd.original.superpoint_data)
        (
            coordinates_a,
            coordinates_o,
            descriptors_a,
            descriptors_o,
        ) = get_coordinates_and_descriptors_from_superglue_data(
            superglue_data, a.superpoint_data, rd.original.superpoint_data
        )
        # look up embedding coordinates via descriptors or old coordinates and calculate a pixel distance between them
        embedded_coordinates = []

        for i in range(len(coordinates_o)):
            for j in range(len(e.extra)):
                if numpy.sqrt(numpy.sum(((coordinates_o[i] - e.extra[j]) ** 2))) < 3:
                    embedded_coordinates.append(e.extra[j])

        # for i in
        print(coordinates_a.shape)
        print(numpy.array(embedded_coordinates).shape)
    # visualization.visualize_matches(a.image, rd.original.image, coordinates_a, coordinates_b)


# df_data = {}
# for stat in statistics:
