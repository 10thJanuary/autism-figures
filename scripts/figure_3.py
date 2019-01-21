import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

from problem import get_train_data, get_test_data

X_train, y_train = get_train_data("..")
X_test, y_test = get_test_data("..")

X_train["type"] = "public set"
X_test["type"] = "private set"
X_train["target"] = y_train
X_test["target"] = y_test
X = pd.concat((X_train, X_test), axis=0)[
    ["participants_site", "participants_sex", "participants_age", "type"]
]
y = np.hstack((y_train, y_test))
X["target"] = y

with sns.plotting_context("poster"):
    fig = plt.figure(figsize=(20, 6))

    # define the figure
    ax_site_1 = fig.add_axes([0.1, 0.1, 0.3, 0.2])
    ax_site_2 = fig.add_axes([0.6, 0.1, 0.3, 0.2])
    ax_subject = fig.add_axes([0.05, 0.5, 0.25, 0.4])
    ax_gender = fig.add_axes([0.39, 0.5, 0.25, 0.4])
    ax_age = fig.add_axes([0.72, 0.5, 0.25, 0.4])

    plt.rcParams["ytick.major.pad"] = 2.5
    plt.rcParams["xtick.major.pad"] = 2.5
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 20

    # target distribution
    train_target = (
        X_train["target"]
        .value_counts()
        .to_frame()
        .T.rename(index={"target": "public\nset"})
    )
    test_target = (
        X_test["target"]
        .value_counts()
        .to_frame()
        .T.rename(index={"target": "private\nset"})
    )
    target = pd.concat([test_target, train_target], axis=0)
    target = target.rename(
        columns={0: "Control", 1: "Autism Spectrum Disorder"}
    )
    target_norm = target.div(target.sum(axis=1), axis=0)

    target_norm.plot.barh(stacked=True, ax=ax_subject)
    sns.despine(offset=5, left=True, ax=ax_subject)
    ax_subject.set_xlim([0, 1])
    ax_subject.set_title("Subjects distribution")
    ax_subject.legend(
        loc="center left", bbox_to_anchor=(0.22, 0.5), prop={"size": 14},
        frameon=False
    )

    n_patients = {}
    for p, idx in zip(ax_subject.patches, [0, 2, 1, 3]):
        ax_subject.annotate(
            "n={:}".format(int(target.values.reshape(-1)[idx])),
            (p.get_x() + (p.get_width() / 2) - 0.1,
            (p.get_y() + p.get_height() / 2.5)),
            size=20,
        )
        n_patients[str(p.get_y())] = p.get_y()

    for (key, value), total_patients in zip(
        n_patients.items(), target.sum(axis=1).values
    ):
        ax_subject.annotate("n={:}".format(total_patients), (0, value + 0.56),
        size=20)

    ax_subject.tick_params(length=6)
    ax_subject.tick_params(axis="y", length=0, labelsize=20)
    ax_subject.set_xticks([0.0, 0.5, 1.0])
    ax_subject.text(
        0.92,
        1,
        "(a)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_subject.transAxes,
    )

    # gender distribution
    train_sex = (
        X_train["participants_sex"]
        .value_counts()
        .to_frame()
        .T.rename(index={"participants_sex": "public\nset"})
    )
    test_sex = (
        X_test["participants_sex"]
        .value_counts()
        .to_frame()
        .T.rename(index={"participants_sex": "private\nset"})
    )
    sex = pd.concat([test_sex, train_sex], axis=0).rename(
        columns={"M": "Male", "F": "Female"}
    )
    sex_norm = sex.div(sex.sum(axis=1), axis=0)

    sex_norm.plot.barh(stacked=True, ax=ax_gender)
    sns.despine(offset=5, left=True, ax=ax_gender)
    ax_gender.set_xlim([0, 1])
    ax_gender.set_title("Sex distribution")
    ax_gender.legend(
        loc="center left", bbox_to_anchor=(0.65, 0.5), prop={"size": 14},
        frameon=False
    )
    n_patients = {}
    for p, idx in zip(ax_gender.patches, [0, 2, 1, 3]):
        ax_gender.annotate(
            "n={:}".format(int(sex.values.reshape(-1)[idx])),
            (p.get_x() + (p.get_width() / 2) - 0.1,
            (p.get_y() + p.get_height() / 2.5)),
            size=20,
        )
        n_patients[str(p.get_y())] = p.get_y()
    for (key, value), total_patients in zip(n_patients.items(),
                                            sex.sum(axis=1).values):
        ax_gender.annotate(
            "n={:}".format(total_patients), (0, value + 0.56), size=20
        )
    ax_gender.tick_params(length=6)
    ax_gender.tick_params(axis="y", length=0, labelsize=20)
    ax_gender.set_xticks([0.0, 0.5, 1.0])
    ax_gender.text(
        0.92,
        1,
        "(b)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_gender.transAxes,
    )

    # age distribution
    X_train["dataset"] = pd.Series(
        ["public set"] * X_train.shape[0], index=X_train.index
    )
    X_test["dataset"] = pd.Series(
        ["private set"] * X_test.shape[0], index=X_test.index
    )
    age_set = pd.concat([X_train, X_test], axis=0)
    age_set[""] = ""
    sns.violinplot(
        x="participants_age",
        y="",
        hue="dataset",
        data=age_set,
        split=True,
        inner="quart",
        ax=ax_age,
        scale="width",
    )

    ax_age.set_xlim([0, 80])
    sns.despine(offset=5, left=True, ax=ax_age)
    ax_age.legend(loc=4, borderaxespad=0.0, prop={"size": 14}, frameon=False)
    ax_age.set_title("Age distribution")
    ax_age.xaxis.set_major_formatter(FormatStrFormatter("%d yr"))
    ax_age.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax_age.tick_params(length=6)
    ax_age.tick_params(axis="y", length=0, labelsize=20)
    ax_age.set_xlabel("")
    ax_age.text(
        0.92,
        1,
        "(c)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_age.transAxes,
    )
    ax_age.set_xticks([0, 40, 80])

    # sites distribution
    train_site = (
        X_train["participants_site"]
        .value_counts()
        .to_frame()
        .T.rename(index={"participants_site": "public\nset"})
    )
    test_site = (
        X_test["participants_site"]
        .value_counts()
        .to_frame()
        .T.rename(index={"participants_site": "private\nset"})
    )
    site = pd.concat([test_site, train_site], axis=0)

    site_abide_1 = site.drop(
        columns=[name for name in site.columns if name > 20]
    )
    site_abide_2 = site.drop(
        columns=[name for name in site.columns if name <= 20]
    ).drop(columns=24)

    site_abide_1 = site_abide_1.div(site_abide_1.sum(axis=1), axis=0)
    site_abide_2 = site_abide_2.div(site_abide_2.sum(axis=1), axis=0)

    site_abide_1.plot.barh(
        stacked=True, ax=ax_site_1, legend=False, colormap=plt.cm.tab20
    )
    sns.despine(offset=5, left=True, ax=ax_site_1)
    ax_site_1.set_xlim([0, 1])
    ax_site_1.set_title("Site distribution for ABIDE I subset")
    ax_site_1.tick_params(length=6)
    ax_site_1.tick_params(axis="y", length=0, labelsize=20)
    ax_site_1.set_xticks([0.0, 0.5, 1.0])
    ax_site_1.text(
        1,
        1,
        "(d)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_site_1.transAxes,
    )

    site_abide_2.plot.barh(
        stacked=True, ax=ax_site_2, legend=False, colormap=plt.cm.tab20
    )
    sns.despine(offset=5, left=True, ax=ax_site_2)
    ax_site_2.set_xlim([0, 1])
    ax_site_2.set_title("Site distribution for ABIDE II subset")
    ax_site_2.tick_params(length=6)
    ax_site_2.tick_params(axis="y", length=0, labelsize=20)
    ax_site_2.set_xticks([0.0, 0.5, 1.0])
    ax_site_2.text(
        1,
        1,
        "(e)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_site_2.transAxes,
    )

    plt.savefig("../figures/dataset.svg")
