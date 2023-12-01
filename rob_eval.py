import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from code import interact

df = pd.read_csv("./data/robustness_statistics.csv")
df = df.drop(columns=["Unnamed: 0"]).reset_index()

df = df.melt(id_vars="index", var_name="variable")
df["artefact_type"] = df["variable"].apply(
    lambda x: ["stretch", "shear", "stretch_shear", "sin"][int(x[-1])]
)
df["variable"] = df["variable"].apply(lambda x: x[:-2])
df = df.pivot_table(index=["index", "artefact_type"], columns="variable")
df.columns = df.columns.droplevel()
df = df.reset_index()
df.columns.name = None

print(df.columns)

df["superglue_confirmation_ratio"] = (
    df["superglue_subset_valid_count"] / df["superglue_subset_count"]
)

df["count_difference_orig_embed"] = np.abs(df["artefact_count"] - df["original_count"])

print(df.drop(columns=["artefact_type"]).mean())

interact(local=locals())

sns.set_context("paper")
sns.set_style("ticks")
sns.set_palette("colorblind")

# c = "superglue_confirmation_ratio"
# ax = sns.boxplot(df, x="artefact_type", y=c, width=0.3)
# ax.set_xlabel("Artefact Type")
# ax.set_ylabel(c.replace("_", " ").title())
# plt.grid(linestyle="dashed")
# plt.tight_layout()
# plt.show()
# for c in df.columns:
#     ax = sns.boxplot(df, x="artefact_type", y=c, width=0.3)
#     ax.set_xlabel("Artefact Type")
#     ax.set_ylabel(c.replace("_", " ").title())
#     plt.grid(linestyle="dashed")
#     plt.tight_layout()
#     plt.show()
