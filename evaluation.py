import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import code
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import (
    RidgeClassifier,
    # RidgeClassifierCV,
    LogisticRegression,
    # LogisticRegressionCV,
)
from sklearn.svm import OneClassSVM, SVC
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
    accuracy_score,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import (
    ClusterCentroids,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report
from pprint import pprint

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

##### USE DATA FROM ALL DATASETS FOR TRAINING


def generate_roc_curves_for_pipe(pipe, dataframe, label_column, pos_label, prefix):
    labels = dataframe[label_column].values
    transformed_data = pipe.fit_transform(dataframe.drop(columns=[label_column]))
    for i in range(transformed_data.shape[1]):
        RocCurveDisplay.from_predictions(
            labels,
            transformed_data[:, i],
            plot_chance_level=True,
            name=pipe.feature_names_in_[i].replace("_", " ").title(),
            pos_label=pos_label,
        )
        plt.savefig(f"./output/{prefix}_RAW_{pipe.feature_names_in_[i]}_ROC.png")


def collect_pipe_averages(pipe, X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    cv = cross_validate(
        pipe,
        X,
        y,
        scoring=[
            "accuracy",
            "f1",
            "recall",
            "precision",
            "roc_auc",
        ],
        cv=7,
    )

    pipe.fit(X_train, y_train)
    rocd = RocCurveDisplay.from_estimator(pipe, X_test, y_test, plot_chance_level=True, name=name)
    return pd.DataFrame(cv), rocd


def lof_eval(X, y, prefix):
    y_hat = lof_pipe.fit_predict(X)
    y_hat = [1 if e < 0 else 0 for e in y_hat]

    stats = {}
    stats["f1"] = [f1_score(y, y_hat)]
    stats["recall"] = [recall_score(y, y_hat)]
    stats["precision"] = [precision_score(y, y_hat)]
    stats["auc"] = [roc_auc_score(y, y_hat)]
    stats["accuracy"] = [accuracy_score(y, y_hat)]
    rocd = RocCurveDisplay.from_predictions(
        y, y_hat, plot_chance_level=True, name="Local Outlier Factor"
    )
    plt.savefig(f"./output/{prefix}_LOF_ROC.png")
    stats = pd.DataFrame(data=stats)
    stats.to_latex(f"./output/{prefix}_LOF_STATS.tex", escape=True)


def if_eval(X, y, prefix):
    if_pipe.fit(X)
    y_hat = if_pipe.predict(X)
    y_hat = [1 if e < 0 else 0 for e in y_hat]

    stats = {}
    stats["f1"] = [f1_score(y, y_hat)]
    stats["recall"] = [recall_score(y, y_hat)]
    stats["precision"] = [precision_score(y, y_hat)]
    stats["auc"] = [roc_auc_score(y, y_hat)]
    stats["accuracy"] = [accuracy_score(y, y_hat)]
    rocd = RocCurveDisplay.from_predictions(
        y, y_hat, plot_chance_level=True, name="Isolation Forest"
    )
    plt.savefig(f"./output/{prefix}_IsolationForest_ROC.png")
    stats = pd.DataFrame(data=stats)
    stats.to_latex(f"./output/{prefix}_IsolationForest_STATS.tex", escape=True)
    pass


# Ignore Images that have non CSNP Tissue Images
X_ip = df_ip.loc[df_ip["is_non_csnp"] == False].drop(columns=["is_non_csnp"])
X_m = df_m.loc[df_m["has_non_csnp"] == False].drop(columns=["has_non_csnp"])


y_ip = X_ip["is_artefact"]
y_m = X_m["has_artefact"]

imbalance_ip = y_ip.mean()
imbalance_m = y_m.mean()

X_train_ip, X_test_ip, y_train_ip, y_test_ip = train_test_split(
    X_ip.drop(columns=["dataset", "is_artefact"]), y_ip
)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_m.drop(columns=["dataset", "has_artefact"]), y_m
)

# How good is each individual metric as a standalone estimator?
raw_threshold_pipe = make_pipeline(StandardScaler(), MinMaxScaler())
# Normal ML Procedures
log_reg_pipe = make_pipeline(StandardScaler(), LogisticRegression())
ridge_class_pipe = make_pipeline(StandardScaler(), RidgeClassifier())
svm_pipe = make_pipeline(StandardScaler(), SVC())

log_reg_pipe_smote = make_pipeline(SMOTE(), StandardScaler(), LogisticRegression())
ridge_class_pipe_smote = make_pipeline(SMOTE(), StandardScaler(), RidgeClassifier())
svm_pipe_smote = make_pipeline(SMOTE(), StandardScaler(), SVC())

oc_svm_pipe = make_pipeline(StandardScaler(), OneClassSVM())
lof_pipe = make_pipeline(StandardScaler(), LocalOutlierFactor())
if_pipe = make_pipeline(StandardScaler(), IsolationForest())


def one_class_svm_eval(X, y, prefix):
    oc_svm_pipe.fit(X.drop(columns=["dataset", "has_artefact"]))
    y_hat_oc_svm = oc_svm_pipe.predict(X.drop(columns=["dataset", "has_artefact"]))
    y_hat_oc_svm = [0 if e > 0 else 1 for e in y_hat_oc_svm]
    oc_svm_stats = {}
    oc_svm_stats["f1"] = [f1_score(y, y_hat_oc_svm)]
    oc_svm_stats["recall"] = [recall_score(y, y_hat_oc_svm)]
    oc_svm_stats["precision"] = [precision_score(y, y_hat_oc_svm)]
    oc_svm_stats["auc"] = [roc_auc_score(y, y_hat_oc_svm)]
    oc_svm_stats["accuracy"] = [accuracy_score(y, y_hat_oc_svm)]
    rocd = RocCurveDisplay.from_predictions(
        y, y_hat_oc_svm, plot_chance_level=True, name="One Class SVM"
    )
    plt.savefig(f"./{prefix}_OneClassSVM_ROC.png")
    stats = pd.DataFrame(data=oc_svm_stats)
    stats.to_latex(f"./output/{prefix}_OneClassSVM_STATS.tex", escape=True)


def eval_pipes(pipes, names, display_names, X, y):
    for pipe, name, display_name in zip(pipes, names, display_names):
        summary_cross_val, rocd = collect_pipe_averages(pipe, X, y, display_name)
        summary_cross_val.loc["mean"] = summary_cross_val.mean()
        summary_cross_val.to_latex(f"./output/{name}_CV.tex", escape=True)
        plt.savefig(f"./output/{name}_ROC.png")


# eval_pipes(
#     [
#         log_reg_pipe,
#         log_reg_pipe_smote,
#         ridge_class_pipe,
#         ridge_class_pipe_smote,
#         svm_pipe,
#         svm_pipe_smote,
#     ],
#     ["M_LogReg", "M_LogReg_SMOTE", "M_RidgeClass", "M_RidgeClass_SMOTE", "M_SVM", "M_SVM_SMOTE"],
#     [
#         "Logistische Regression",
#         "Logistische Regression (SMOTE)",
#         "Ridge Classifier",
#         "Ridge Classifier (SMOTE)",
#         "SVM",
#         "SVM (SMOTE)",
#     ],
#     X_m.drop(columns=["dataset", "has_artefact"]),
#     y_m,
# )


# eval_pipes(
#     [
#         log_reg_pipe,
#         log_reg_pipe_smote,
#         ridge_class_pipe,
#         ridge_class_pipe_smote,
#         svm_pipe,
#         svm_pipe_smote,
#     ],
#     [
#         "IP_LogReg",
#         "IP_LogReg_SMOTE",
#         "IP_RidgeClass",
#         "IP_RidgeClass_SMOTE",
#         "IP_SVM",
#         "IP_SVIP_SMOTE",
#     ],
#     [
#         "Logistische Regression",
#         "Logistische Regression (SMOTE)",
#         "Ridge Classifier",
#         "Ridge Classifier (SMOTE)",
#         "SVM",
#         "SVM (SMOTE)",
#     ],
#     X_ip.drop(columns=["dataset", "is_artefact"]),
#     y_ip,
# )

# generate_roc_curves_for_pipe(
#     raw_threshold_pipe, X_m.drop(columns=["dataset"]), "has_artefact", False, "M"
# )
# generate_roc_curves_for_pipe(
#     raw_threshold_pipe, X_ip.drop(columns=["dataset"]), "is_artefact", False, "IP"
# )

# one_class_svm_eval(X_m.drop(columns=["dataset", "has_artefact"]), y_m, "M")
# one_class_svm_eval(X_ip.drop(columns=["dataset", "is_artefact"]), y_ip, "IP")

# if_eval(X_m.drop(columns=["dataset", "has_artefact"]), y_m, "M")
# if_eval(X_ip.drop(columns=["dataset", "is_artefact"]), y_ip, "IP")

lof_eval(X_m.drop(columns=["dataset", "has_artefact"]), y_m, "M")
lof_eval(X_ip.drop(columns=["dataset", "is_artefact"]), y_ip, "IP")
