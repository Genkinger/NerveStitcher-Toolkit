import superpoint
import superglue
import cv2
import torch
import numpy
import config
from os import listdir
from os.path import join, splitext

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
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0 for path in paths]


def preprocess_images(images: numpy.ndarray):
    average = numpy.average(images, axis=0)
    return images / average * numpy.average(average)
