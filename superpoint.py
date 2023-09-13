from pathlib import Path
import torch
from torch import nn
from matplotlib import pyplot as plt
import config


def non_maximum_suppression(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    def __init__(
        self, descriptor_dimensions: int = config.DESCRIPTOR_DIMENSIONS, device: str = config.DEVICE
    ):
        super().__init__()
        self.device = device

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, descriptor_dimensions, kernel_size=1, stride=1, padding=0)

    def load_weights(self, path: str):
        load_result = torch.load(path)
        self.load_state_dict(load_result)
        print("[INFO]: Loaded SuperPoint Weights.")

    def forward(self, input: torch.Tensor, nms_radius: int = config.NON_MAXIMUM_SUPPRESSION_RADIUS):
        x = self.relu(self.conv1a(input))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        scores_batch_size, scores_channels, scores_height, scores_width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(
            scores_batch_size, scores_height, scores_width, 8, 8
        )
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            scores_batch_size, scores_height * 8, scores_width * 8
        )
        scores = non_maximum_suppression(scores, nms_radius)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        input_batch_size, input_channels, input_height, input_width = input.shape
        identity_transform = torch.Tensor([[1, 0, 0], [0, 1, 0]])
        identity_transform = torch.stack([identity_transform for i in range(input_batch_size)])
        sampling_grid = torch.nn.functional.affine_grid(
            identity_transform, [input_batch_size, input_channels, input_height, input_width]
        ).to(self.device)

        descriptors_upsampled = torch.nn.functional.grid_sample(
            descriptors, sampling_grid, mode="bilinear", align_corners=False
        )
        descriptors_upsampled = torch.nn.functional.normalize(descriptors_upsampled)

        return scores, descriptors_upsampled
