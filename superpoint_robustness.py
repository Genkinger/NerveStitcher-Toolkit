import nervestitcher
import image_transform
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy
from superpoint import SuperPointData
import code
import visualization
import pickle


@dataclass
class ImageData:
    image: numpy.ndarray
    superpoint_data: SuperPointData


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
    embedding = numpy.full((height, width), None, numpy.object_)
    for c, s, d in zip(coordinates, scores, descriptors):
        embedding[c[1], c[0]] = (s, d)
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
    for c in coordinates:
        s, d = embedding[c[1], c[0]]
        scores.append(s)
        descriptors.append(d)
    scores = numpy.array(scores)
    descriptors = numpy.array(descriptors)
    superpoint_data = SuperPointData(coordinates, scores, descriptors, width, height)
    image = numpy.zeros_like(embedding, numpy.float32)
    image[coordinates[:, 1], coordinates[:, 0]] = 1
    return ImageData(image, superpoint_data)


def extract_robustness_data(images, scale=0.75, movements=[]):
    robustness_data = []
    for image in images:
        rd = RobustnessData(None, [], [])
        rd.original = ImageData(image, nervestitcher.superpoint(image))

        embedding = create_embedding(rd.original.superpoint_data)

        for movement in movements:
            artefacted_image = image_transform.apply_artefact(
                image, int(image.shape[1] * scale), int(image.shape[0] * scale), movement
            )
            superpoint_data = nervestitcher.superpoint(artefacted_image)
            rd.artefacts.append(ImageData(artefacted_image, superpoint_data))

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

robustness_data = extract_robustness_data(images, scale=0.75, movements=movements)


with open("./data/robustness_data.pkl", "w+") as pklfile:
    pickle.dump(robustness_data, pklfile)

# # TODO: Now we need to see how many points of the retransformed coordinates are contained within the transformed data. After that we need to compare the descriptors first by vector distance, then by SuperGlue
statistics = []
for rd in robustness_data:
    statistics_local = []
    for a, e in zip(rd.artefacts, rd.embeddings):
        distances = calculate_distance_matrix(
            a.superpoint_data.coordinates, e.superpoint_data.coordinates
        )
        # Consider any interestpoints more than max_distance pixels apart as non-matching
        max_distance = 3
        distances = distances <= max_distance
        # row without entries -> point in artefacts not in embeddings
        # column without entries -> point in embeddings not in artefacts
        points_in_artefact_len, points_in_embedding_len = distances.shape
        points_in_embedding_not_in_artefact = numpy.argwhere(
            numpy.count_nonzero(distances, axis=0) == 0
        )
        points_in_artefact_not_in_embedding = numpy.argwhere(
            numpy.count_nonzero(distances, axis=1) == 0
        )
        points_in_embedding_not_in_artefact_len = len(points_in_embedding_not_in_artefact)
        points_in_artefact_not_in_embedding_len = len(points_in_artefact_not_in_embedding)
        hit_rate = points_in_embedding_len / (
            points_in_artefact_len - points_in_artefact_not_in_embedding_len
        )
        statistics_local.append(hit_rate)
    statistics.append(statistics_local)

statistics = numpy.array(statistics)
stat_t = numpy.transpose(statistics)

for series in stat_t:
    plt.plot(series)
plt.show()
