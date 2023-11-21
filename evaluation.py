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


hue = "is_artefact"
for col in df_m.columns:
    # g = kde_grid(df_ip, "dataset", "is_artefact", col)
    ax = sns.boxplot(data=df_ip, y="dataset", hue=hue, x=col, fliersize=2)
    ax.set_title(col.replace("_", " ").title())
    ax.set_ylabel("")
    # ax.set_xlabel(col.replace("_", " ").title())
    ax.set_xlabel("")
    plt.subplots_adjust(left=0.2)
    ax.get_figure().legend(title=hue.replace("_", " ").title(), loc="lower right")
    plt.grid(linestyle="dashed")
    sns.despine(trim=True)
    ax.get_figure().savefig(f"./ip_{col}_box.png")
    plt.cla()
