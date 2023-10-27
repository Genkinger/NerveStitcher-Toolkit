import cv2
import numpy


def apply_gaussian_noise(mean, var, image):
    height, width = image.shape
    random = numpy.random.normal(mean, var**0.5, (height, width))
    return numpy.clip(random + image, 0, 1)


def apply_artefact(
    source_image: numpy.ndarray, width: int, height: int, movement, out_type=numpy.float32
) -> numpy.ndarray:
    """Performs an artificial scan of source_image with movement artefacts.

    :param source_image: specifies the source image of which the artificial scan is performed
        the source image needs to have dimensions big enough so that an offset of max(movement)
        does not produce indices that are out of bounds.
    :param width: specifies the output width
    :param height: specifies the output height
    :param movement: a callable returning a x,y offset tuple for a given input t.
        t represents a normalized time coordinate between 0 and 1 inclusiveo
    :return: scan result
    """
    source_height, source_width = source_image.shape
    destination_image = numpy.full((width, height), None)
    x_offset = (source_width - width) // 2
    y_offset = (source_height - height) // 2
    t = 0.0
    increment = 1.0 / (width * height)
    for y in range(height):
        for x in range(width):
            x_movement, y_movement = movement(t)
            x_movement = int(x_movement * x_offset)
            y_movement = int(y_movement * y_offset)
            t += increment
            destination_image[y, x] = source_image[
                y + y_offset + y_movement, x + x_offset + x_movement
            ]
    return destination_image.astype(out_type)


def generate_checkerboard_image(width: int, height: int, size: int) -> numpy.ndarray:
    image = numpy.zeros((width, height))
    cells_x, cells_y = width // size, height // size
    for i in range(cells_x):
        for j in range(cells_y):
            image[j * cells_y : (j + 1) * cells_y, i * cells_x : (i + 1) * cells_x] = (i + j) % 2
    return image


def generate_fake_nerve_image():
    pass
