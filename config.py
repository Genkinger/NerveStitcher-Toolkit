import torch.cuda
import logging

DESCRIPTOR_DIMENSIONS: int = 256
NON_MAXIMUM_SUPPRESSION_RADIUS: int = 4
SUPPORTED_FILE_EXTENSIONS: list[str] = ["png", "tif", "jpg", "jpeg"]
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_THRESHOLD: float = 0.005
KEYPOINT_ENCODER_LAYERS: list[int] = [32, 64, 128, 256]
GNN_LAYERS: list[str] = ["self", "cross"] * 9
SINKHORN_ITERATIONS: int = 100
MATCHING_THRESHOLD: float = 0.8
LOG_LEVEL = logging.DEBUG
