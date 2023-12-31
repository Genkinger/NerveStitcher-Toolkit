import numpy
import nervestitcher as ns
import pickle
import cv2
import config
import log
from typing import NamedTuple
from superpoint import SuperPointData
from superglue import SuperGlueData
import code

logger = log.get_logger(__name__)


class MatchData(NamedTuple):
    superglue_data: SuperGlueData
    coordinates_a: numpy.ndarray
    coordinates_b: numpy.ndarray
    affine_transform: numpy.ndarray
    inliers: numpy.ndarray


def generate_cos2_weight_image(width: int, height: int):
    image = numpy.zeros((height, width))
    for y in range(height):
        for x in range(width):
            image[y, x] = (
                numpy.sin(numpy.pi * (x - 0.5) / width) * numpy.sin(numpy.pi * (y - 0.5) / height)
            ) ** 2
    return image


def save_interest_point_data(path, superpoint_data_list):
    with open(path, "wb+") as picklefile:
        pickle.dump(superpoint_data_list, picklefile)


def load_interest_point_data(path) -> list[SuperPointData]:
    with open(path, "rb") as picklefile:
        superpoint_data_list = pickle.load(picklefile)
        return superpoint_data_list


def generate_interest_point_data(images, threshold=config.SCORE_THRESHOLD):
    superpoint_data_list = []
    for i, image in enumerate(images):
        logger.info(f"Generating interestpoint data {i+1}/{len(images)}")
        superpoint_data = ns.superpoint(image, score_threshold=threshold)
        superpoint_data_list.append(superpoint_data)
    return superpoint_data_list


def generate_raw_match_data(
    superpoint_data_list: list[SuperPointData],
    diagonals=0,
    threshold=config.MATCHING_THRESHOLD,
):
    n = len(superpoint_data_list)
    count = diagonals if diagonals > 0 else n
    raw_match_matrix = numpy.full((n, n), None)

    for i in range(n):
        for j in range(i + 1, min(i + 1 + count, n)):
            logger.info(f"Matching {i} against {j}")
            superglue_data: SuperGlueData = ns.superglue(
                superpoint_data_list[i], superpoint_data_list[j], threshold
            )

            valid = superglue_data.indices_a > -1
            coordinates_a = superpoint_data_list[i].coordinates[valid]
            coordinates_b = superpoint_data_list[j].coordinates[superglue_data.indices_a[valid]]
            transform, inliers = None, None
            try:
                transform, inliers = cv2.estimateAffinePartial2D(coordinates_a, coordinates_b)
            except:
                logger.warn(f"Something went wrong when calculating transform between {i} and {j}")
            match_data = MatchData(superglue_data, coordinates_a, coordinates_b, transform, inliers)
            raw_match_matrix[i][j] = match_data
    return raw_match_matrix


def vsplit_coordinates(coordinates, splits: int, height: int):
    parts = []
    slice_height = height // splits
    for i in range(splits):
        parts.append([])
        for c in coordinates:
            if i * slice_height < c[1] and c[1] < (i + 1) * slice_height:
                parts[i].append(c)
    parts = [numpy.array(p) for p in parts]
    return parts


def save_raw_match_matrix(path, raw_match_matrix):
    with open(path, "wb+") as picklefile:
        pickle.dump(raw_match_matrix, picklefile)


def load_raw_match_matrix(path) -> list[MatchData]:
    with open(path, "rb") as picklefile:
        raw_match_matrix = pickle.load(picklefile)
        return raw_match_matrix


def get_match_translation_matrix_from_raw_match_matrix(raw_match_matrix):
    pass


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
