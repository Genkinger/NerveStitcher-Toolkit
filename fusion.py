import numpy
import nervestitcher as ns
import torch
import pickle
import cv2
import config


def generate_cos2_weight_image(width: int, height: int):
    image = numpy.zeros((height, width))
    for y in range(height):
        for x in range(width):
            image[y, x] = (
                numpy.sin(numpy.pi * (x - 0.5) / width) * numpy.sin(numpy.pi * (y - 0.5) / height)
            ) ** 2
    return image


def fuse(image_list, absolute_positions):
    pass


def save_interest_point_data(path, coordinates, scores, descriptors):
    with open(path, "wb+") as picklefile:
        pickle.dump(coordinates, picklefile)
        pickle.dump(scores, picklefile)
        pickle.dump(descriptors, picklefile)


def load_interest_point_data(path):
    coordinates, scores, descriptors = None, None, None
    with open(path, "rb") as picklefile:
        coordinates = pickle.load(picklefile)
        scores = pickle.load(picklefile)
        descriptors = pickle.load(picklefile)
    return coordinates, scores, descriptors


def generate_interest_point_data(greyscale_image_list, threshold=config.SCORE_THRESHOLD):
    image_tensors = [torch.from_numpy(image).float()[None][None] for image in greyscale_image_list]

    coordinates = []
    scores = []
    descriptors = []
    for image_tensor in image_tensors:
        c, s, d = ns.superpoint(image_tensor, score_threshold=threshold)
        coordinates.extend(c)
        scores.extend(s)
        descriptors.extend(d)

    return coordinates, scores, descriptors


def generate_interest_point_data_chunked(greyscale_image_list, chunk_size=50):
    image_tensors = torch.stack(
        [torch.from_numpy(image).float()[None] for image in greyscale_image_list]
    )

    coordinates = []
    scores = []
    descriptors = []
    for chunk in ns.chunked(image_tensors, chunk_size):
        c, s, d = ns.superpoint(chunk)
        coordinates.extend(c)
        scores.extend(s)
        descriptors.extend(d)

    return coordinates, scores, descriptors


# def generate_matching_data_temporal(greyscale_image_list, coordinates, scores, descriptors):
#     image_count = len(greyscale_image_list)
#     index_pairs = list(zip(range(image_count - 1), range(1, image_count)))
#     matching_coordinates = []
#     for index_pair in index_pairs:
#         image_a = torch.from_numpy(greyscale_image_list[index_pair[0]]).float()[None][None]
#         image_b = torch.from_numpy(greyscale_image_list[index_pair[1]]).float()[None][None]
#         indices_a, indices_b, scores_a, scores_b = ns.superglue(
#             image_a,
#             coordinates[index_pair[0]][None],
#             scores[index_pair[0]][None],
#             descriptors[index_pair[0]][None],
#             image_b,
#             coordinates[index_pair[1]][None],
#             scores[index_pair[1]][None],
#             descriptors[index_pair[1]][None],
#             ns.config.MATCHING_THRESHOLD,
#         )
#         valid = indices_a[0] > -1
#         coordinates_a = coordinates[index_pair[0]][valid]
#         coordinates_b = coordinates[index_pair[1]][indices_a[0][valid]]
#         matching_coordinates.append((coordinates_a.cpu().numpy(), coordinates_b.cpu().numpy()))
#     return matching_coordinates, index_pairs


def generate_local_transforms_from_coordinate_pair_list(coordinate_pair_list):
    transforms = []
    for coordinate_pair in coordinate_pair_list:
        transform = cv2.estimateAffinePartial2D(coordinate_pair[0], coordinate_pair[1])
        transforms.append(transform[0])
    return transforms


def generate_matching_data_n_vs_n(greyscale_image_list, coordinates, scores, descriptors):
    image_count = len(greyscale_image_list)

    adjacency_matrix = numpy.full((image_count, image_count), None)

    for i in range(image_count):
        for j in range(i + 1, image_count):
            print(f"Matching {i} against {j}...")
            image_a = greyscale_image_list[i]
            image_b = greyscale_image_list[j]
            image_a = torch.from_numpy(image_a).float()[None][None]
            image_b = torch.from_numpy(image_b).float()[None][None]
            indices_a, indices_b, scores_a, scores_b = ns.superglue(
                image_a,
                coordinates[i][None],
                scores[i][None],
                descriptors[i][None],
                image_b,
                coordinates[j][None],
                scores[j][None],
                descriptors[j][None],
                config.MATCHING_THRESHOLD,
            )
            valid = indices_a[0] > -1
            coordinates_a = coordinates[i][valid]
            coordinates_b = coordinates[j][indices_a[0][valid]]
            transform, _ = cv2.estimateAffinePartial2D(
                coordinates_a.cpu().numpy(), coordinates_b.cpu().numpy()
            )
            if transform is not None:
                adjacency_matrix[i][j] = numpy.array([transform[0][2], transform[1][2]])
    return adjacency_matrix


def save_adjacency_matrix(path, adjacency_matrix):
    with open(path, "wb+") as picklefile:
        pickle.dump(adjacency_matrix, picklefile)


def load_adjacency_matrix(path):
    adjacency_matrix = None
    with open(path, "rb") as picklefile:
        adjacency_matrix = pickle.load(picklefile)
    return adjacency_matrix
