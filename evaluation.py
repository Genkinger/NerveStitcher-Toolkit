import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import code
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import (
    RidgeClassifier,
    RidgeClassifierCV,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    RocCurveDisplay,
    roc_curve,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_recall_curve,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE, ADASYN

sns.set_style("ticks")
sns.set_context("paper")
sns.set_palette("colorblind")

df_ip = pd.read_csv("./data/ipdata_full.csv")
df_m = pd.read_csv("./data/matchdata_full.csv")


def kde_grid(d, row, hue, value):
    g = sns.FacetGrid(data=d, row=row, hue=hue, aspect=10, height=0.75, legend_out=False)
    g.map(
        sns.kdeplot,
        value,
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=0.75,
        multiple="stack",
    )

    def label(x, color, label):
        ax = plt.gca()
        if label == "True":
            ax.text(
                0,
                0.2,
                x.iloc[0],
                color="gray",
                fontweight="bold",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    g.map(label, row)
    # xmin = d.groupby([row, hue])[value].quantile(0.005).min()
    # xmax = d.groupby([row, hue])[value].quantile(0.99).max()
    # print(xmin)
    # print(xmax)
    # for ax in g.figure.axes:
    #     ax.set_ylim(0, 1)

    g.figure.subplots_adjust(hspace=0.25)
    g.set_titles("")
    g.set(xlabel=value.replace("_", " ").title())
    g.set(yticks=[], ylabel="")
    g.add_legend(title=hue.replace("_", " ").title(), loc="lower right", frameon=True)
    g.despine(left=True)
    return g


# code.interact(local=locals())
# hue = "has_artefact"
# df = df_m
# for col in df.columns:
#     # g = kde_grid(df_ip, "dataset", "is_artefact", col)
#     ax = sns.boxplot(data=df, x="dataset", hue=hue, y=col, fliersize=2)
#     ax.set_title(col.replace("_", " ").title())
#     plt.xticks(rotation=45)
#     # ax.set_label("")
#     # ax.set_xlabel(col.replace("_", " ").title())
#     ax.set_xlabel("")
#     ax.set_ylabel("")
#     plt.subplots_adjust(bottom=0.2)
#     ax.legend(title=hue.replace("_", " ").title())
#     plt.grid(linestyle="dashed")
#     # sns.despine(trim=True)
#     ax.get_figure().savefig(f"./m_{col}_box.png")
#     # plt.show()
#     plt.clf()


###################################################
# Classifiers | ROC CURVES | AUC SCORES #
###################################################

# Thresholding?
# Binary Ridge Regression
# LOC
# IF
# OCSVM
# LOGREG

# Probleme: Sehr starke Imbalance -> SMOTE?
# 1.) Test Train Split
# 2.) FIT
# 3.) Predict
# 4.) ROC Curve + AUC + Prec + Recall + F1 + Confusion
# 5.) Selbes aber mit PCA und Haendisch

csnp_ip = df_ip.loc[df_ip["is_non_csnp"] == False]
csnp_m = df_m.loc[df_m["has_non_csnp"] == False]

valid_count = csnp_m["valid_count"].values
transform_inlier_ratio = csnp_m["transform_inlier_ratio"].values
labels = csnp_m["has_artefact"].values

# pipe = make_pipeline(StandardScaler(), MinMaxScaler())
pipe = make_pipeline(StandardScaler())
valid_count_transformed = pipe.fit_transform(valid_count.reshape(-1, 1))
# fpr, tpr, thresholds = roc_curve(labels, valid_count)
# RocCurveDisplay.from_predictions(labels, valid_count_transformed, pos_label=1)
RocCurveDisplay.from_predictions(labels, valid_count_transformed, pos_label=False)
plt.show()
