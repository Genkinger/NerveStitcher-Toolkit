import fusion
import nervestitcher


images = nervestitcher.load_images_in_directory("/home/leah/Datasets/EGT7_001-A_4_snp")
images = nervestitcher.preprocess_images(images[:100])

coordinates, scores, descriptors = fusion.load_interest_point_data(
    "./data/EGT7_001-A_4_snp_IP_0.005.pkl"
)
coordinates = coordinates[:100]
scores = scores[:100]
descriptors = descriptors[:100]


adjacency = fusion.generate_matching_data_n_vs_n(images, coordinates, scores, descriptors)
fusion.save_adjacency_matrix("./data/EGT7_001-A_4_snp_match_adjacency.pkl", adjacency)
