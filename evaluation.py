import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import code

sns.set_style("ticks")
sns.set_context("paper")
sns.set_palette("colorblind")


df_ip = pd.read_csv("./data/ipdata_full.csv")
df_m = pd.read_csv("./data/matchdata_full.csv")


def kde_grid(data, row, hue):
    g = sns.FacetGrid(data, row=row, hue=hue, aspect=10, height=1, legend_out=True)
    g.map(
        sns.kdeplot,
        "mean_score_log",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=0.6,
    )

    def label(x, color, label):
        ax = plt.gca()
        if label == "True":
            ax.text(
                0.05,
                0.2,
                x.iloc[0],
                color="Black",
                fontweight="bold",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    g.map(label, "dataset")
    g.figure.subplots_adjust(hspace=0.25)
    g.set_titles("")
    g.set(xlabel=row.replace("_", " ").capitalize())
    g.set(yticks=[], ylabel="")
    g.add_legend(title="Artefact", loc="lower right", frameon=True)
    g.despine(left=True)


kde_grid(df_ip, "mean_score", "is_artefact")
plt.show()
