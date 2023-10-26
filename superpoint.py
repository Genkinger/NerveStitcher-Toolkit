import torch
from torch import nn
import config
from typing import NamedTuple
import numpy


class SuperPointData(NamedTuple):
    coordinates: numpy.ndarray
    scores: numpy.ndarray
    descriptors: numpy.ndarray
    width: int
    height: int


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

    def __init__(self, descriptor_dimensions: int = config.DESCRIPTOR_DIMENSIONS):
        super().__init__()
        self.descriptor_dimensions = descriptor_dimensions

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
        self.convDb = nn.Conv2d(c5, self.descriptor_dimensions, kernel_size=1, stride=1, padding=0)
        self.to(config.DEVICE)

    def load_weights(self, path: str):
        load_result = torch.load(path)
        self.load_state_dict(load_result)
        print("[INFO]: Loaded SuperPoint Weights.")

    def forward(
        self,
        input: numpy.ndarray,
        score_threshold: float = config.SCORE_THRESHOLD,
        nms_radius: int = config.NON_MAXIMUM_SUPPRESSION_RADIUS,
    ):
        assert (
            len(input.shape) == 2
        ), "Only single grayscale image inputs are supported at the moment!"
        input = torch.from_numpy(input[None, None, :]).float()

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

        coordinates = [torch.nonzero(score > score_threshold).flip(dims=[1]) for score in scores]
        scores_keep = [scores[i, coord[:, 1], coord[:, 0]] for i, coord in enumerate(coordinates)]

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Sample the descriptors here
        # NOTE(Leah): this is my own implementation. It would be wise to compare it to the reference implementation

        input_batch_size, input_channels, input_height, input_width = input.shape
        descriptor_samples = []
        for coords, descriptor in zip(coordinates, descriptors):
            sample = torch.nn.functional.grid_sample(
                descriptor[None],
                (
                    coords / torch.tensor([input_width, input_height], device=config.DEVICE) * 2.0
                    - 1.0
                ).view(1, 1, -1, 2),
                mode="bilinear",
            )
            sample = sample.reshape(1, self.descriptor_dimensions, -1)
            sample = torch.nn.functional.normalize(sample, p=2, dim=1)
            descriptor_samples.append(sample[0])

        # NOTE(Leah): same sampling but inefficient
        # input_batch_size, input_channels, input_height, input_width = input.shape
        # identity_transform = torch.Tensor([[1, 0, 0], [0, 1, 0]])
        # identity_transform = torch.stack([identity_transform for i in range(input_batch_size)])
        # sampling_grid = torch.nn.functional.affine_grid(
        #     identity_transform, [input_batch_size, input_channels, input_height, input_width]
        # ).to(self.device)

        # descriptors_upsampled = torch.nn.functional.grid_sample(
        #     descriptors, sampling_grid, mode="bilinear", align_corners=False
        # )
        # descriptors_upsampled = torch.nn.functional.normalize(descriptors_upsampled)

        # descriptor_samples = []
        # for coords, descriptor in zip(coordinates, descriptors_upsampled):
        #     descriptor_samples.append(descriptor[:, coords[:, 1], coords[:, 0]])

        return SuperPointData(
            coordinates[0].cpu().numpy(),
            scores_keep[0].cpu().numpy(),
            descriptor_samples[0].cpu().numpy(),
            input_width,
            input_height,
        )
