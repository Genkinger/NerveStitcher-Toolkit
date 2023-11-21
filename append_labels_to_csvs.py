import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def per_image_mask_to_image_pair_mask(mask):
    pair_mask = [False for _ in range(len(mask) - 1)]
    for i, e in enumerate(mask):
        if e == True:
            if i > 0:
                pair_mask[i - 1] = True
            if i < len(pair_mask):
                pair_mask[i] = True
    return pair_mask


def append_labels(current_dataset_name):
    f = open(f"./data/labels/{current_dataset_name}_snp.pkl", "rb")
    mask = pickle.load(f)
    f.close()
    artefact_mask = [True if e == 1 else False for e in mask]
    non_csnp_mask = [True if e == 2 else False for e in mask]

    df_ip = pd.read_csv(f"./data/ipdata/{current_dataset_name}.csv")
    df_m = pd.read_csv(f"./data/matchdata/{current_dataset_name}.csv")

    df_ip["is_non_csnp"] = non_csnp_mask
    df_ip["is_artefact"] = artefact_mask
    df_m["has_artefact"] = per_image_mask_to_image_pair_mask(artefact_mask)
    df_m["has_non_csnp"] = per_image_mask_to_image_pair_mask(non_csnp_mask)

    df_ip.to_csv(f"./data/ipdata/{current_dataset_name}_labeled.csv", index=False)
    df_m.to_csv(f"./data/matchdata/{current_dataset_name}_labeled.csv", index=False)


datasets = [
    "EGT6_001_A_2",
    "EGT6_005_G_4",
    "EGT6_008_S_3",
    "EGT6_009_S_2",
    "EGT7_001_A_3",
    "EGT7_001_A_4",
]

for d in datasets:
    append_labels(d)

# SET STYLE
sns.set_style("whitegrid")
sns.set_palette("colorblind")
sns.set_context("paper")
