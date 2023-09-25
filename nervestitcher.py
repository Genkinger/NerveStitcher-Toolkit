import code
import superpoint
import superglue
import cv2
import torch
import numpy
import config
from os import listdir
from os.path import join, splitext
from matplotlib import pyplot as plt
from itertools import islice
from dataclasses import dataclass

torch.set_grad_enabled(False)
superpoint = superpoint.SuperPoint()
superpoint.load_weights("./weights/superpoint.pth")
superglue = superglue.SuperGlue()
superglue.load_weights("./weights/superglue.pth")


def load_images_in_directory(
    directory: str, extensions: list[str] = config.SUPPORTED_FILE_EXTENSIONS
):
    paths = [
        join(directory, path) for path in listdir(directory) if splitext(path)[1][1:] in extensions
    ]
    paths.sort()
    return numpy.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0 for path in paths])


def preprocess_images(images: numpy.ndarray):
    average = numpy.average(images, axis=0)
    return (images / average) * numpy.average(average)


def chunked(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def viz_coordinates(image, coordinates, scores):
    img = image.cpu().numpy().squeeze()
    coords = coordinates.cpu().numpy().squeeze()
    scrs = scores.cpu().numpy().squeeze()

    plt.imshow(img, cmap="Greys_r")
    plt.scatter(coords[:, 0], coords[:, 1], s=scores * 5, c="magenta")
    plt.show()

    code.interact(local=locals())


def viz_matches(image_a, image_b, coordinates_a, coordinates_b):
    pass


def combine_images_naive(previous_image, next_image, matrix, next_image_offset, current_size):
    h, w = next_image.shape
    next_image_offset.x -= matrix[0][2]
    next_image_offset.y -= matrix[1][2]
    previous_image_offset = Point()

    if next_image_offset.x < 0:
        current_size.x += abs(next_image_offset.x)
        previous_image_offset.x = -next_image_offset.x
        next_image_offset.x = 0

    if next_image_offset.y < 0:
        current_size.y += abs(next_image_offset.y)
        previous_image_offset.y = -next_image_offset.y
        next_image_offset.y = 0

    if next_image_offset.x + w > current_size.x:
        current_size.x = next_image_offset.x + w
    if next_image_offset.y + h > current_size.y:
        current_size.y = next_image_offset.y + h

    new_image = numpy.zeros((int(current_size.y), int(current_size.x)), numpy.uint8)

    slice_for_previous_image_y = slice(
        int(previous_image_offset.y), int(previous_image_offset.y + previous_image.shape[0])
    )
    slice_for_previous_image_x = slice(
        int(previous_image_offset.x), int(previous_image_offset.x + previous_image.shape[1])
    )
    slice_for_next_image_y = slice(
        int(next_image_offset.y), int(next_image_offset.y + next_image.shape[0])
    )
    slice_for_next_image_x = slice(
        int(next_image_offset.x), int(next_image_offset.x + next_image.shape[1])
    )

    new_image[slice_for_previous_image_y, slice_for_previous_image_x] = previous_image
    new_image[slice_for_next_image_y, slice_for_next_image_x] = next_image

    return new_image


def do_stitching_naive(image_pairs, width, height):
    current_size = Point(width, height)
    next_image_offset = Point(0.0, 0.0)
    previous_image = image_pairs[0][0] * 255.0

    for i, (image_a, image_b) in enumerate(image_pairs):
        input_a = torch.from_numpy(image_a).float()[None][None]
        input_b = torch.from_numpy(image_b).float()[None][None]
        coordinates_a, scores_a, descriptors_a = superpoint(input_a, config.SCORE_THRESHOLD)
        coordinates_b, scores_b, descriptors_b = superpoint(input_b, config.SCORE_THRESHOLD)

        idx0, idx1, score0, score1 = superglue(
            input_a,
            coordinates_a[0][None],
            scores_a[0][None],
            descriptors_a[0][None],
            input_b,
            coordinates_b[0][None],
            scores_b[0][None],
            descriptors_b[0][None],
            config.MATCHING_THRESHOLD,
        )

        valid = idx0[0] > -1
        mkpts0 = coordinates_a[0][valid].cpu().numpy()
        mkpts1 = coordinates_b[0][idx0[0][valid]].cpu().numpy()

        if not mkpts0.shape[0] > 0 or not mkpts1.shape[0] > 0:
            print(f"[INFO]: No Keypoints extracted in iteration {i}")
            print(f"[INFO]: Continuing!")
            previous_image = image_b
            next_image_offset = Point()
            current_size = Point(previous_image.shape[1], previous_image.shape[0])
            continue

        matrix, _ = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
        print(matrix)
        if matrix is None:
            print(f"[INFO]: Unable to estimate transform")
            print(f"[INFO]: Continuing!")
            previous_image = image_b
            next_image_offset = Point()
            current_size = Point(previous_image.shape[1], previous_image.shape[0])
            continue

        previous_image = combine_images_naive(
            previous_image, image_b * 255.0, matrix, next_image_offset, current_size
        )
        cv2.imwrite(join("./output", f"intermediate_{i}.jpg"), previous_image)


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __abs__(self):
        return Point(abs(self.x), abs(self.y))
