import tomllib
import nervestitcher
import numpy
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import OPTICS, DBSCAN, Birch
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import make_pipeline


import data.artefacts_EGT7_001_A_3_snp
import data.artefacts_EGT7_001_A_4_snp
import pandas as pd


optics_pipe = make_pipeline(RobustScaler(), OPTICS())
dbscan_pipe = make_pipeline(RobustScaler(), DBSCAN())
# hdbscan_pipe = make_pipeline(RobustScaler(), HDBSCAN())
birch_pipe = make_pipeline(RobustScaler(), Birch())
oneclasssvm_pipe = make_pipeline(RobustScaler(), OneClassSVM(nu=0.13))
lof_pipe = make_pipeline(RobustScaler(), LocalOutlierFactor())
isolationforest_pipe = make_pipeline(RobustScaler(), IsolationForest())
logreg_pipe = make_pipeline(RobustScaler(), LogisticRegression())

optics_pipe_pca = make_pipeline(RobustScaler(), PCA(), OPTICS())
dbscan_pipe_pca = make_pipeline(RobustScaler(), PCA(), DBSCAN())
# hdbscan_pipe_pca = make_pipeline(RobustScaler(), PCA(), HDBSCAN())
birch_pipe_pca = make_pipeline(RobustScaler(), PCA(), Birch())
oneclasssvm_pipe_pca = make_pipeline(RobustScaler(), PCA(3), OneClassSVM(nu=0.13))
lof_pipe_pca = make_pipeline(RobustScaler(), PCA(), LocalOutlierFactor())
isolationforest_pipe_pca = make_pipeline(RobustScaler(), PCA(), IsolationForest())
logreg_pipe_pca = make_pipeline(RobustScaler(), PCA(), LogisticRegression())


ip_data_a3 = pd.read_csv("./data/ipdata/EGT7_001-A_3.csv")
ip_data_a3["is_artefact"] = data.artefacts_EGT7_001_A_3_snp.EGT7_001_A_3_snp_artefact_mask
ip_data_a3_no_artefacts = ip_data_a3.loc[ip_data_a3["is_artefact"] == 0].drop(
    columns=["is_artefact"]
)
print(ip_data_a3_no_artefacts)
ip_data_a3_just_artefacts = ip_data_a3.loc[ip_data_a3["is_artefact"] == 1].drop(
    columns=["is_artefact"]
)
print(ip_data_a3_just_artefacts)
oneclasssvm_pipe_pca.fit(ip_data_a3_no_artefacts)
predicted_labels = oneclasssvm_pipe_pca.predict(ip_data_a3_just_artefacts)
print(predicted_labels)
