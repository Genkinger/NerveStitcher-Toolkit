import fusion
from superpoint import SuperPointData
from superglue import SuperGlueData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tomllib
from sklearn import cluster
from sklearn import decomposition
import config
from fusion import MatchData
import code
import scipy.stats


def sort_ip_data_by_score(data: SuperPointData):
    indices = np.flip(data.scores.argsort())
    scores = data.scores[indices]
    coordinates = data.coordinates[indices, :]
    descriptors = data.descriptors[:, indices]
    return SuperPointData(coordinates, scores, descriptors, data.width, data.height)


def calculate_statistics_for_ip_data(list_of_data: list[SuperPointData]):
    output = {
        "mean_score": [],
        "median_score": [],
        "std_score": [],
        "var_score": [],
        "min_score": [],
        "max_score": [],
        "skew_score": [],
        "kurtosis_score": [],
        "ip_count": [],
    }
    for data in list_of_data:
        output["mean_score"].append(data.scores.mean())
        output["median_score"].append(np.median(data.scores))
        output["min_score"].append(data.scores.min())
        output["max_score"].append(data.scores.max())
        output["ip_count"].append(len(data.scores))
        output["std_score"].append(data.scores.std())
        output["var_score"].append(data.scores.var())
        output["skew_score"].append(scipy.stats.skew(data.scores))
        output["kurtosis_score"].append(scipy.stats.kurtosis(data.scores))

    return pd.DataFrame(data=output).reset_index(drop=True)


def calculate_statistics_for_match_data(
    list_of_data: list[MatchData], threshold: float = config.MATCHING_THRESHOLD
):
    output = {
        "mean_score": [],
        "median_score": [],
        "std_score": [],
        "var_score": [],
        "min_score": [],
        "max_score": [],
        "skew_score": [],
        "kurtosis_score": [],
        "mean_score_log": [],
        "median_score_log": [],
        "std_score_log": [],
        "var_score_log": [],
        "min_score_log": [],
        "max_score_log": [],
        "skew_score_log": [],
        "kurtosis_score_log": [],
        "transform_inlier_ratio": [],
        "valid_count": [],
    }
    for data in list_of_data:
        scores_log = data.superglue_data.scores[:-1, :-1]
        scores = np.exp(scores_log)
        output["mean_score"].append(scores.mean())
        output["median_score"].append(np.median(scores))
        output["std_score"].append(scores.std())
        output["var_score"].append(scores.var())
        output["valid_count"].append(np.count_nonzero(scores > threshold))
        output["min_score"].append(np.min(scores))
        output["max_score"].append(np.max(scores))
        output["skew_score"].append(scipy.stats.skew(scores, axis=None))
        output["kurtosis_score"].append(scipy.stats.kurtosis(scores, axis=None))
        output["transform_inlier_ratio"].append(
            np.count_nonzero(data.inliers) / len(data.coordinates_a)
        )
        output["mean_score_log"].append(scores_log.mean())
        output["median_score_log"].append(np.median(scores_log))
        output["std_score_log"].append(scores_log.std())
        output["var_score_log"].append(scores_log.var())
        output["min_score_log"].append(np.min(scores_log))
        output["max_score_log"].append(np.max(scores_log))
        output["skew_score_log"].append(scipy.stats.skew(scores_log, axis=None))
        output["kurtosis_score_log"].append(scipy.stats.kurtosis(scores_log, axis=None))

    return pd.DataFrame(data=output).reset_index(drop=True)


# ip_data = fusion.load_interest_point_data("./data/a3_ip_0.005.pkl")
# match_data = fusion.load_raw_match_matrix("./data/a3_match.pkl")
# match_data = [match_data[i, i + 1] for i in range(len(match_data) - 1)]


# ip_stats = calculate_statistics_for_ip_data(ip_data)
# match_stats = calculate_statistics_for_match_data(match_data)

# ip_stats.to_csv("./data/a3_ip_stats.csv")
# match_stats.to_csv("./data/a3_match_stats.csv")

ip_paths = [
    "./data/ipdata/EGT6_001-A_2.pkl",
    "./data/ipdata/EGT6_005-G_4.pkl",
    "./data/ipdata/EGT6_008-S_3.pkl",
    "./data/ipdata/EGT6_009-S_2.pkl",
    "./data/ipdata/EGT7_001-A_3.pkl",
    "./data/ipdata/EGT7_001-A_4.pkl",
]
match_paths = [
    "./data/matchdata/EGT6_001-A_2.pkl",
    "./data/matchdata/EGT6_005-G_4.pkl",
    "./data/matchdata/EGT6_008-S_3.pkl",
    "./data/matchdata/EGT6_009-S_2.pkl",
    "./data/matchdata/EGT7_001-A_3.pkl",
    "./data/matchdata/EGT7_001-A_4.pkl",
]

for p in ip_paths:
    ipdata = fusion.load_interest_point_data(p)
    stats = calculate_statistics_for_ip_data(ipdata)
    stats.to_csv(p.replace("pkl", "csv").replace("-", "_"), index=False)

for p in match_paths:
    match_data = fusion.load_raw_match_matrix(p)
    diagonal_1 = [match_data[i, i + 1] for i in range(len(match_data) - 1)]
    stats = calculate_statistics_for_match_data(diagonal_1)
    stats.to_csv(p.replace("pkl", "csv").replace("-", "_"), index=False)
