import nervestitcher as ns


images = ns.load_images_in_directory("/home/leah/Dataset/EGT7_001-A_4_snp")

scores, descriptors = ns.superpoint()
