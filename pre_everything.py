import nervestitcher as ns
import os.path
import os
import cv2
import numpy

datasets = "/home/leah/Datasets"
output_folder = "/home/leah/Datasets/Preprocessed"
folders = [(os.path.join(datasets, p), p) for p in os.listdir("/home/leah/Datasets/")]

for path, folder in folders:
    images = ns.load_images_in_directory(path)
    images = ns.preprocess_images(images)
    for i, image in enumerate(images):
        image *= 255
        image = image.astype(numpy.uint8)
        output_image_folder = os.path.join(
            output_folder,
            folder,
        )
        os.makedirs(output_image_folder, exist_ok=True)
        output_image_path = os.path.join(output_image_folder, f"{folder}_{i:04d}.tif")
        cv2.imwrite(output_image_path, image)
