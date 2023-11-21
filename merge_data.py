import numpy as np
import pandas as pd

datasets = [
    "EGT6_001_A_2",
    "EGT6_005_G_4",
    "EGT6_008_S_3",
    "EGT6_009_S_2",
    "EGT7_001_A_3",
    "EGT7_001_A_4",
]
csnp_images = []
csnp_pairs = []
for d in datasets:
    df_ip = pd.read_csv(f"./data/ipdata/{d}_labeled.csv")
    df_m = pd.read_csv(f"./data/matchdata/{d}_labeled.csv")
    df_ip["dataset"] = [d] * len(df_ip.index)
    df_m["dataset"] = [d] * len(df_m.index)
    csnp_images.append(df_ip)
    csnp_pairs.append(df_m)

df_ip = pd.concat(csnp_images)
df_m = pd.concat(csnp_pairs)

df_ip.to_csv("./data/ipdata_full.csv", index=False)
df_m.to_csv("./data/matchdata_full.csv", index=False)
