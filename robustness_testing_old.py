import image_transform
import nervestitcher
import fusion
import scipy.ndimage
from matplotlib import pyplot as plt
import cv2
import numpy
import torch
from visualization import image_grid, image_grid_with_coordinates


def movement(t):
    x_offset = 0.4
    if t < x_offset:
        return 0, 0
    return numpy.sin((t - x_offset) * 3 * numpy.pi) * 0.6, 0


def make_artefact_pair(image, scan_w, scan_h, movement):
    h, w = image.shape
    artefacted_image = image_transform.apply_artefact(image, scan_w, scan_h, movement)
    return (
        image[(h - scan_h) // 2 : -(h - scan_h) // 2, (w - scan_w) // 2 : -(w - scan_w) // 2],
        artefacted_image,
    )


def generate_superpoint_testing_data(image, scan_w, scan_h, movement):
    h, w = image.shape
    original_image_scan, artefacted_image_scan = make_artefact_pair(image, scan_w, scan_h, movement)
    c_a, _, _ = nervestitcher.superpoint(
        torch.from_numpy(original_image_scan).float()[None][None], 0.005
    )
    c_b, _, _ = nervestitcher.superpoint(
        torch.from_numpy(artefacted_image_scan).float()[None][None], 0.005
    )

    coordinate_embedding = numpy.zeros_like(image)
    for x, y in c_a[0]:
        coordinate_embedding[
            (h - scan_h) // 2 : -(h - scan_h) // 2, (w - scan_w) // 2 : -(w - scan_w) // 2
        ][y, x] = 1

    transformed_coordinate_image = image_transform.apply_artefact(
        coordinate_embedding, scan_w, scan_h, movement
    )
    post_transform_coordinates = numpy.flip(numpy.argwhere(transformed_coordinate_image), axis=1)
    return original_image_scan, artefacted_image_scan, c_a[0], c_b[0], post_transform_coordinates


def compare_coordinate_lists(coordinates_a, coordinates_b):
    dist_matrix = numpy.zeros((len(coordinates_a), len(coordinates_b)))
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix[0])):
            dist_matrix[i, j] = numpy.sum((coordinates_a[i] - coordinates_b[j]) ** 2) ** 0.5
    minima = numpy.argmin(dist_matrix, axis=1)
    min_matrix = numpy.zeros_like(dist_matrix)
    for i in range(len(minima)):
        min_matrix[i, minima[i]] = 1
    return image_grid([dist_matrix, min_matrix], 1, 2)


images = nervestitcher.load_images_in_directory("/home/leah/Datasets/EGT7_001-A_4_snp")
images = nervestitcher.preprocess_images(images)

(
    original_image_scan,
    artefacted_image_scan,
    original_coordinates,
    artefact_coordinates,
    original_coordinates_retransformed,
) = generate_superpoint_testing_data(images[10], 300, 300, movement)

fig, ax = image_grid_with_coordinates(
    [original_image_scan, artefacted_image_scan, artefacted_image_scan],
    [original_coordinates, artefact_coordinates, original_coordinates_retransformed],
    1,
    3,
    4,
    ["magenta", "magenta", "cyan"],
)

fig2, _ = compare_coordinate_lists(
    artefact_coordinates.cpu().numpy(), original_coordinates_retransformed
)
fig.show()
fig2.show()
plt.show()
