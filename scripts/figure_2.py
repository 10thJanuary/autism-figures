import os
import random

import numpy as np
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import joblib

from nilearn import datasets, input_data, plotting, image
from sklearn.svm import SVC

import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sp
from nilearn import datasets, input_data, plotting, image
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score

import matplotlib.colors as colors
from problem import get_train_data, get_test_data

X_train, y_train = get_train_data("..")
X_test, y_test = get_test_data("..")

all_submissions = [
    "abethe_functional_blast",
    "amicie_functional_blast",
    "ayoub.ghriss_functional_blast",
    "mk_functional_blast",
    "nguigui_functional_blast",
    "pearrr_functional_blast",
    "Slasnista_functional_blast",
    "vzantedeschi_functional_blast",
    "wwwwmmmm_functional_blast",
]

results = {"0%": [], "25%": [], "50%": [], "75%": []}
for submission_name in all_submissions:
    path_prediction = os.path.join(
        "..", "submissions", submission_name, "training_output",
        "y_pred_test.joblib"
    )

    y_pred = joblib.load(path_prediction)

    def compute_score(y_true_test, y_pred_test):
        return roc_auc_score(y_true_test, y_pred_test[:, 1])

    for idx, percentage_blasting in enumerate(results.keys()):
        if percentage_blasting != "0%":
            results[percentage_blasting].append(
                compute_score(y_test, y_pred[idx - 1])
            )

    path_prediction = path_prediction.replace("_blast", "")
    path_prediction = path_prediction.replace(".joblib", ".npy")

    results["0%"].append(compute_score(y_test, np.load(path_prediction)))

df = pd.DataFrame(
    results,
    index=["Participants #{}".format(i + 1)
           for i in range(len(all_submissions))],
)
df_boxplot = df.stack().to_frame().reset_index()

all_linestyles = [""]
all_markerstyles = [".", "^", "s", "X"]

styles = [
    random.choice(all_markerstyles) + random.choice(all_linestyles)
    for i in range(df.shape[0])
]

with sns.plotting_context("poster"):
    fig = plt.figure(figsize=(8, 3))
    ax_box_plot = fig.add_axes([0.63, 0.2, 0.3, 0.58])
    ax_glass_brain = fig.add_axes([0.0, 0, 0.45, 1])

    # Seaborn creates too much padding here
    plt.rcParams["ytick.major.pad"] = 2.5
    plt.rcParams["xtick.major.pad"] = 2.5
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 16

    sns.boxplot(
        x="level_1",
        y=0,
        data=df_boxplot,
        whis=10.0,
        ax=ax_box_plot,
        width=0.4,
        palette=sns.cubehelix_palette(4, reverse=True),
    )
    df.T.plot(
        ax=ax_box_plot,
        markersize=11,
        alpha=0.4,
        marker=".",
        linestyle="",
        color="magenta",
        zorder=10,
    )
    for i in range(2):
        ax_box_plot.axvspan(2 * i + 0.5, 2 * i + 1.5, color=".9", zorder=0)
    ax_box_plot.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_box_plot.set_ylabel("ROC-AUC", size=18)
    ax_box_plot.set_xlabel("Amount of regions removed", size=18)
    ax_box_plot.legend_.remove()
    sns.despine(offset=10, bottom=True)
    ax_box_plot.set_yticks([0.60, 0.69, 0.78])
    ax_box_plot.set_ylim([0.60, 0.78])
    ax_box_plot.tick_params(length=6)
    ax_box_plot.tick_params(axis="x", length=0, labelsize=16)
    ax_box_plot.text(
        1.05,
        0.93,
        "(b)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_box_plot.transAxes,
    )


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


from nilearn import image, datasets

mean_img = image.load_img("../data/statistical_map.nii")

masker = input_data.NiftiMasker(
    mask_img=image.resample_to_img(
        datasets.load_mni152_brain_mask(), mean_img, interpolation="nearest"
    )
)

# plot the image
masked_img = masker.fit_transform(mean_img)
unmasked_img = masker.inverse_transform(masked_img)
display = plotting.plot_glass_brain(
    unmasked_img,
    threshold=0,
    cmap=plt.cm.viridis,
    colorbar=False,
    display_mode="xz",
    axes=ax_glass_brain,
)
# ax_glass_brain.text(1, 1, '(a)', fontweight='bold')
ax1 = fig.add_axes([0.05, 0.12, 0.4, 0.03])
cmap = truncate_colormap(plt.cm.viridis, minval=0.5)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(
    ax1, cmap=cmap, norm=norm, orientation="horizontal")
cb1.set_ticks([0, 1])
ax1.text(0.12, -3.5, "Ranking of regions", size=18)
ax_glass_brain.text(
    0,
    0.75,
    "(a)",
    fontweight="bold",
    size=16,
    va="center",
    transform=ax_glass_brain.transAxes,
)
fig.suptitle(
    "Importance of regions in functional MRI", fontweight="bold", size=20
)
plt.savefig("../figures/second_figure.svg")
