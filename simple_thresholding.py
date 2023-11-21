import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data.artefacts_EGT7_001_A_3_snp import EGT7_001_A_3_snp_artefact_mask as a3_mask
from data.artefacts_EGT7_001_A_4_snp import EGT7_001_A_4_snp_artefact_mask as a4_mask
from sklearn.metrics import auc
import code


def per_image_artefact_mask_to_image_pair_mask(artefact_mask):
    # es werden nur 1,2 2,3 3,4 etc beruecksichtigt:
    # bei n bildern gibt es n-1 matches
    # ist bild i von artefakt betroffen so ist jeder match der i beinhaltet betroffen i-1,i und i,i+1
    match_mask = [0 for _ in range(len(artefact_mask) - 1)]
    for i, mask in enumerate(artefact_mask):
        if mask == 1:
            if i > 0:
                match_mask[i - 1] = 1
            if i < len(match_mask):
                match_mask[i] = 1
    return match_mask


def roc(df, column, artefact_mask):
    tprs = []
    fprs = []
    p = np.array(artefact_mask).sum()
    n = len(artefact_mask) - p
    steps = 100
    thresholds = np.linspace(df[column].min(), df[column].max(), steps)
    for threshold in thresholds:
        indices = df.loc[df[column] < threshold].index.tolist()
        tp = 0
        fp = 0
        for i in indices:
            if artefact_mask[i] == 1:
                tp += 1
            if artefact_mask[i] == 0:
                fp += 1
        tprs.append(tp / p)
        fprs.append(fp / n)
    return np.array(tprs), np.array(fprs)


df = pd.read_csv("./data/matchdata/EGT7_001-A_4.csv")


tprs, fprs = roc(
    df,
    "mean_score",
    per_image_artefact_mask_to_image_pair_mask(a4_mask),
)


def plot_roc(tprs, fprs, title):
    print(f"AUC: {auc(fprs,tprs)}")
    plt.rc("grid", linestyle="dashed", color="lightgray")
    plt.title(f"ROC - {title}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.plot(fprs, tprs)
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.plot([0, 1])
    plt.grid(True)
    auclabel = plt.text(0.82, 0.03, f"AUC: {auc(fprs,tprs):.3}")
    auclabel.set_bbox(dict(facecolor="white"))
    plt.show()


plot_roc(tprs, fprs, "transform_inlier_ratio")
