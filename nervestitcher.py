import superpoint
import superglue
import cv2
import torch
import numpy
import config
from os import listdir
from os.path import join, splitext
from itertools import islice

torch.set_grad_enabled(False)
superpoint = superpoint.SuperPoint()
superpoint.load_weights("./weights/superpoint.pth")
superglue = superglue.SuperGlue()
superglue.load_weights("./weights/superglue.pth")


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
    return (images / average) * numpy.average(average)


def chunked(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch
