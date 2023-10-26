from copy import deepcopy
import torch
from torch import nn
import config
from typing import NamedTuple
import numpy
from superpoint import SuperPointData


class SuperGlueData(NamedTuple):
    scores: numpy.ndarray
    indices_a: numpy.ndarray
    indices_b: numpy.ndarray
    scores_a: numpy.ndarray
    scores_b: numpy.ndarray


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, width, height):
    """Normalize keypoints locations based on image image_shape"""
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            l(x).view(batch_dim, self.dim, self.num_heads, -1)
            for l, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]
        )
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """

    def __init__(
        self,
        descriptor_dimensions: int = config.DESCRIPTOR_DIMENSIONS,
        keypoint_encoder_layers: list[int] = config.KEYPOINT_ENCODER_LAYERS,
        gnn_layers: list[str] = config.GNN_LAYERS,
        sinkhorn_iterations: int = config.SINKHORN_ITERATIONS,
    ):
        super().__init__()
        self.sinkhorn_iterations = sinkhorn_iterations
        self.descriptor_dimensions = descriptor_dimensions

        self.kenc = KeypointEncoder(self.descriptor_dimensions, keypoint_encoder_layers)

        self.gnn = AttentionalGNN(self.descriptor_dimensions, gnn_layers)

        self.final_proj = nn.Conv1d(
            self.descriptor_dimensions, self.descriptor_dimensions, kernel_size=1, bias=True
        )

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)
        self.to(config.DEVICE)

    def load_weights(self, path: str):
        load_result = torch.load(path)
        self.load_state_dict(load_result)
        print("[INFO]: Loaded SuperGlue Weights.")

    def forward(
        self,
        superpoint_data_a: SuperPointData,
        superpoint_data_b: SuperPointData,
        matching_threshold: float = config.MATCHING_THRESHOLD,
    ):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        """Assume only one image per batch, different options not supported at this time!"""

        desc0 = torch.from_numpy(superpoint_data_a.descriptors[None])
        desc1 = torch.from_numpy(superpoint_data_b.descriptors[None])
        kpts0 = torch.from_numpy(superpoint_data_a.coordinates[None])
        kpts1 = torch.from_numpy(superpoint_data_b.coordinates[None])

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, superpoint_data_a.width, superpoint_data_a.height)
        kpts1 = normalize_keypoints(kpts1, superpoint_data_b.width, superpoint_data_b.height)

        # Keypoint MLP encoder.
        enc0 = self.kenc(kpts0, torch.from_numpy(superpoint_data_a.scores[None]))
        enc1 = self.kenc(kpts1, torch.from_numpy(superpoint_data_b.scores[None]))
        desc0 = desc0 + enc0
        desc1 = desc1 + enc1

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / self.descriptor_dimensions**0.5
        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iters=self.sinkhorn_iterations)

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)

        indices0, indices1 = max0.indices, max1.indices

        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)

        zero = scores.new_tensor(0)

        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)

        valid0 = mutual0 & (mscores0 > matching_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)

        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return SuperGlueData(
            scores[0].cpu().numpy(),
            indices0[0].cpu().numpy(),
            indices1[0].cpu().numpy(),
            mscores0[0].cpu().numpy(),
            mscores1[0].cpu().numpy(),
        )
