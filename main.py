import nervestitcher as ns

images = ns.load_images_in_directory("/home/leah/Dataset/snippet")
images = ns.preprocess_images(images)

image_pairs = list(zip(images[:-1], images[1:]))
ns.do_stitching_naive(image_pairs, 384, 384)
