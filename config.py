import torch.cuda



DESCRIPTOR_DIMENSIONS = 256
NON_MAXIMUM_SUPPRESSION_RADIUS = 4
SUPPORTED_FILE_EXTENSIONS = ["png","tif","jpg","jpeg"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"