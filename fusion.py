import numpy
import nervestitcher as ns
import torch
import pickle
import cv2
import config
import log

logger = log.get_logger(__name__)


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
        logger.info(f"Generating interestpoint data {i+1}/{len(image_tensors)}")
        c, s, d = ns.superpoint(image_tensor, score_threshold=threshold)
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


def match_images(
    image_a,
    image_b,
    coordinates_a,
    coordinates_b,
    scores_a,
    scores_b,
    descriptors_a,
    descriptors_b,
    threshold=config.MATCHING_THRESHOLD,
):
    image_a = torch.from_numpy(image_a).float()[None][None]
    image_b = torch.from_numpy(image_b).float()[None][None]
    indices_a, indices_b, scores_a, scores_b = ns.superglue(
        image_a,
        coordinates_a[None],
        scores_a[None],
        descriptors_a[None],
        image_b,
        coordinates_b[None],
        scores_b[None],
        descriptors_b[None],
        threshold,
    )
    valid = indices_a[0] > -1
    coordinates_a_out = coordinates_a[valid]
    coordinates_b_out = coordinates_b[indices_a[0][valid]]
    return coordinates_a_out, coordinates_b_out, scores_a, scores_b


def generate_raw_match_matrix(
    greyscale_image_list,
    coordinates,
    scores,
    descriptors,
    diagonals=0,
    threshold=config.MATCHING_THRESHOLD,
):
    image_count = len(greyscale_image_list)
    count = diagonals if diagonals > 0 else image_count
    raw_match_matrix = numpy.full((image_count, image_count), None)

    for i in range(image_count):
        for j in range(i + 1, min(i + 1 + count, image_count)):
            logger.info(f"Matching {i} against {j}")
            coordinates_a, coordinates_b, scores_a, scores_b = match_images(
                greyscale_image_list[i],
                greyscale_image_list[j],
                coordinates[i],
                coordinates[j],
                scores[i],
                scores[j],
                descriptors[i],
                descriptors[j],
            )

            raw_match_matrix[i][j] = (coordinates_a, coordinates_b, scores_a, scores_b)
    return raw_match_matrix


def save_raw_match_matrix(path, raw_match_matrix):
    with open(path, "wb+") as picklefile:
        pickle.dump(raw_match_matrix, picklefile)


def load_raw_match_matrix(path):
    raw_match_matrix = None
    with open(path, "rb") as picklefile:
        raw_match_matrix = pickle.load(picklefile)
    return raw_match_matrix


def get_match_translation_matrix_from_raw_match_matrix(raw_match_matrix):
    match_translation_matrix = numpy.full_like(raw_match_matrix, None)
    for i in range(len(raw_match_matrix)):
        for j in range(len(raw_match_matrix[0])):
            data = raw_match_matrix[i, j]
            if data is None:
                continue
            logger.info(f"estimating partial affine transform between {i} and {j}")
            transform, _ = cv2.estimateAffinePartial2D(data[0].cpu().numpy(), data[1].cpu().numpy())
            if transform is None:
                continue
            match_translation_matrix[i, j] = numpy.array([transform[0, 2], transform[1, 2]])
    return match_translation_matrix


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


def solve_match_translation_matrix(match_translation_matrix, image_width=384, image_height=384):
    rows, columns = match_translation_matrix.shape
    coeffs = [numpy.zeros(columns)]
    coeffs[0][0] = 1
    bxs, bys = [0], [0]
    for i in range(rows):
        for j in range(columns):
            element = match_translation_matrix[i, j]
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
    print(len(bxs))
    logger.info("Solving Linear Equations...")
    positions_x, _, _, _ = numpy.linalg.lstsq(numpy.stack(coeffs), bxs, rcond=-1)
    positions_y, _, _, _ = numpy.linalg.lstsq(numpy.stack(coeffs), bys, rcond=-1)
    logger.info("Solution Found")
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
    # weight_accumulator = numpy.zeros_like(canvas)
    # weight_image = generate_cos2_weight_image(image_list[0].shape[1], image_list[0].shape[0])
    for i in range(len(image_list)):
        offset = (positions_x[i], positions_y[i])
        place_image_at_offset(canvas, image_list[i], offset)
        # add_image_at_offset(canvas, image_list[i] * weight_image, offset)
        # add_image_at_offset(weight_accumulator, weight_image, offset)
    # weight_accumulator[numpy.nonzero(weight_accumulator == 0)] = 1.0
    # canvas /= weight_accumulator
    return canvas
