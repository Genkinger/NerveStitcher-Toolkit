import numpy
import fusion
import nervestitcher
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, SpectralClustering
from sklearn.preprocessing import StandardScaler, Normalizer
import code
import tomllib


def get_local_sliding_windows_padded(array, window_size=51):
    val = numpy.pad(array, window_size // 2, mode="symmetric")
    return sliding_window_view(val, window_size)


ip_data = fusion.load_interest_point_data("./data/a3_ip_0.005.pkl")
m_data = fusion.load_raw_match_matrix("./data/a3_match.pkl")
images = nervestitcher.load_images_in_directory("/home/leah/Datasets/EGT7_001-A_3_snp")
artefacts = {}
with open("./data/artefacts.toml", "rb") as tomlfile:
    artefacts = tomllib.load(tomlfile)

scores_pre = [d.scores[:-1, :-1] for d in m_data[numpy.nonzero(m_data)]]
scores = [numpy.exp(d.scores[:-1, :-1]) for d in m_data[numpy.nonzero(m_data)]]

scores_pre_mean = numpy.array([s.mean() for s in scores_pre])
scores_pre_mean_mean = get_local_sliding_windows_padded(scores_pre_mean).mean(axis=1)
scores_pre_mean_std = get_local_sliding_windows_padded(scores_pre_mean).std(axis=1)
scores_pre_mean_z = (scores_pre_mean - scores_pre_mean_mean) / scores_pre_mean_std

scores_mean = numpy.array([s.mean() for s in scores])
scores_mean_mean = get_local_sliding_windows_padded(scores_mean).mean(axis=1)
scores_mean_std = get_local_sliding_windows_padded(scores_mean).std(axis=1)
scores_mean_z = (scores_mean - scores_mean_mean) / scores_mean_std

scores_valid_count = numpy.array([numpy.count_nonzero(s > 0.80) for s in scores])
scores_valid_count_mean = get_local_sliding_windows_padded(scores_valid_count).mean(axis=1)
scores_valid_count_std = get_local_sliding_windows_padded(scores_valid_count).std(axis=1)
scores_valid_count_z = (scores_valid_count - scores_valid_count_mean) / scores_valid_count_std

scores_dim_count_a = numpy.array([s.shape[0] for s in scores])
scores_dim_count_b = numpy.array([s.shape[1] for s in scores])
scores_dim_count_avg = (scores_dim_count_a + scores_dim_count_b) / 2

percent_valid_count = scores_valid_count / scores_dim_count_a
pvc_mean = get_local_sliding_windows_padded(percent_valid_count).mean(axis=1)
pvc_std = get_local_sliding_windows_padded(percent_valid_count).std(axis=1)
pvc_z = (percent_valid_count - pvc_mean) / pvc_std


def plot(data, title, x_label, y_label, colors, width, height):
    plt.rcParams["figure.figsize"] = (width, height)
    for d, c in zip(data, colors):
        plt.plot(d, c=c)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.show()


def bar(data, title, x_label, y_label, colors, width, height):
    plt.rcParams["figure.figsize"] = (width, height)
    for i, (d, c) in enumerate(zip(data, colors)):
        plt.bar(range(len(d)), d, color=c)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.grid()
    plt.show()


def scatter(data, title, x_label, y_label, colors, width, height):
    plt.rcParams["figure.figsize"] = (width, height)
    for d, c in zip(data, colors):
        plt.scatter(d[0], d[1], c=c, marker="x")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.show()


# for idx in artefacts["EGT7_001-A_3_snp"]["artefacts"]:
#     plt.imshow(images[idx - 1])
#     plt.show()


code.interact(local=locals())
