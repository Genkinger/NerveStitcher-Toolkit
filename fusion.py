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
    for i, image_tensor in enumerate(image_tensors):
        print(f"Generating interestpoint data {i}/{len(image_tensors)}")
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


def generate_local_transforms_from_coordinate_pair_list(coordinate_pair_list):
    transforms = []
    for coordinate_pair in coordinate_pair_list:
        transform = cv2.estimateAffinePartial2D(coordinate_pair[0], coordinate_pair[1])
        transforms.append(transform[0])
    return transforms


def generate_adjacency_matrix_full(
    greyscale_image_list, coordinates, scores, descriptors, threshold=config.MATCHING_THRESHOLD
):
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
                threshold,
            )
            valid = indices_a[0] > -1
            coordinates_a = coordinates[i][valid]
            coordinates_b = coordinates[j][indices_a[0][valid]]

            dists = numpy.linalg.norm((coordinates_b - coordinates_a), ord=2, axis=1)
            mean = numpy.mean(dists)
            stddev = numpy.std(dists)
            z = (dists - mean) / stddev
            coordinates_a = numpy.delete(coordinates_a, z > 0, axis=0)
            coordinates_b = numpy.delete(coordinates_b, z > 0, axis=0)

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


def add_image_at_offset(base, image, offset):
    base[
        int(offset[1]) : int(offset[1]) + image.shape[0],
        int(offset[0]) : int(offset[0]) + image.shape[1],
    ] += image


def place_image_at_offset(base, image, offset):
    base[
        int(offset[1]) : int(offset[1]) + image.shape[0],
        int(offset[0]) : int(offset[0]) + image.shape[1],
    ] = image


def solve_adjacency_matrix(adjacency_matrix, image_width=384, image_height=384):
    rows, columns = adjacency_matrix.shape
    coeffs = [numpy.zeros(columns)]
    coeffs[0][0] = 1
    bxs, bys = [0], [0]
    for i in range(rows):
        for j in range(columns):
            element = adjacency_matrix[i, j]
            if element is None:
                continue
            element = element.astype(numpy.float32)
            if numpy.count_nonzero(element > min(image_height, image_width)) > 0:
                continue
            coeff_row = numpy.zeros(columns)
            coeff_row[i] = 1
            coeff_row[j] = -1
            coeffs.append(coeff_row)
            bxs.append(element[0])
            bys.append(element[1])

    positions_x, _, _, _ = numpy.linalg.lstsq(numpy.stack(coeffs), bxs, rcond=None)
    positions_y, _, _, _ = numpy.linalg.lstsq(numpy.stack(coeffs), bys, rcond=None)
    return positions_x, positions_y


def fuse(image_list, positions_x, positions_y):
    min_x = numpy.min(positions_x)
    min_y = numpy.min(positions_y)
    max_x = numpy.max(positions_x)
    max_y = numpy.max(positions_y)
    width = int(max_x - min_x + image_list[0].shape[1])
    height = int(max_y - min_y + image_list[0].shape[0])
    positions_x -= min_x
    positions_y -= min_y
    canvas = numpy.zeros((height, width))
    weight_accumulator = numpy.zeros_like(canvas)
    weight_image = generate_cos2_weight_image(image_list[0].shape[1], image_list[0].shape[0])
    for i in range(len(image_list)):
        offset = (positions_x[i], positions_y[i])
        # place_image_at_offset(canvas, image_list[i], offset)
        add_image_at_offset(canvas, image_list[i] * weight_image, offset)
        add_image_at_offset(weight_accumulator, weight_image, offset)
    weight_accumulator[numpy.nonzero(weight_accumulator == 0)] = 1.0
    canvas /= weight_accumulator
    return canvas
