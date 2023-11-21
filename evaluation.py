import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import code

datasets = [
    "EGT6_001_A_2",
    "EGT6_005_G_4",
    "EGT6_008_S_3",
    "EGT6_009_S_2",
    "EGT7_001_A_3",
    "EGT7_001_A_4",
]

# SET STYLE
sns.set_style("ticks")
# sns.set_style("whitegrid")
sns.set_palette("colorblind")
sns.set_context("paper")


# other_stats = {}
csnp_images = []
for d in datasets:
    df_ip = pd.read_csv(f"./data/ipdata/{d}_labeled.csv")
    # df_m = pd.read_csv(f"./data/matchdata/{d}_labeled.csv")
    df_ip["dataset"] = [d] * len(df_ip.index)
    csnp_images.append(df_ip)

    # print(csnp_images.groupby("is_artefact").agg(["mean", "median", "max", "min", "skew"]))

    # artefacted_images = csnp_images.loc[csnp_images["is_artefact"] == True]
    # non_artefacted_images = csnp_images.loc[csnp_images["is_artefact"] == False]
    # non_csnp_images = df_ip.loc[df_ip["is_non_csnp"] == True]

    # means = csnp_images.mean()
    # medians = csnp_images.median()
    # std = csnp_images.std()
    # var = csnp_images.var()
    # print(means)
    # print(medians)
    # print(std)
    # print(var)
    # non_csnp_count = len(non_csnp_images.index)
    # artefacted_count = len(artefacted_images.index)
    # csnp_count = len(csnp_images.index)

    # SACHEN DRAWN

    # other_stats[d] = {
    #     "count": len(df_ip.index),
    #     "artefact_count": artefacted_count,
    #     "csnp_count": csnp_count,
    #     "non_csnp_count": non_csnp_count,
    #     "non_csnp_ratio": non_csnp_count / len(df_ip.index),
    #     "artefact_ratio": artefacted_count / csnp_count,
    # }

    # df = pd.DataFrame(data=other_stats)
    # df.to_csv("./data/datasets_other_stats.csv")
    # df.to_latex("../bachelor-thesis/dataset_info.tex")
# print(collection)
df = pd.concat(csnp_images)
print(df)
# df = pd.DataFrame(data={"idx": collection.keys(), "frames": collection.values()})
# print(df)
# code.interact(local=locals())
sns.violinplot(
    data=df, x="dataset", y="ip_count", hue="is_artefact", split=True, inner="quart", fill=False
)
sns.despine(trim=True)
# sns.despine()
plt.show()
