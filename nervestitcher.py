import superpoint
import cv2
import torch
import matplotlib.pyplot as plt
import code
import numpy
import config
from os import listdir
from os.path import join, splitext, expanduser
from itertools import islice


superpoint = superpoint.SuperPoint()


def split_coordinates(coordinates: torch.Tensor):
    coord_split = []
    last_index = 0
    for i in range(1, len(coordinates) - 1):
        if coordinates[i - 1][0] != coordinates[i][0]:
            coord_split.append(coordinates[last_index:i])
            last_index = i
    coord_split.append(coordinates[last_index:])
    return coord_split


def load_images_in_directory(
    directory: str, extensions: list[str] = config.SUPPORTED_FILE_EXTENSIONS
):
    return numpy.array(
        [
            cv2.imread(join(directory, path), cv2.IMREAD_GRAYSCALE) / 255.0
            for path in listdir(directory)
            if splitext(path)[1][1:] in extensions
        ]
    )


def preprocess_images(images: numpy.ndarray):
    average = numpy.average(images, axis=0)
    return images / average


def chunked(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch
