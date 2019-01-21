import os
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
import joblib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

from scipy import interp
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

from utils import load_train_test_prediction
from utils import compute_roc_auc_score
from problem import get_test_data
from utils import load_train_test_prediction_learning_curve

with sns.plotting_context("poster"):
    fig = plt.figure(figsize=(20, 5))
    ax_box_plot_2 = fig.add_axes([0.48, 0.70, 0.13, 0.22])
    ax_box_plot_1 = fig.add_axes([0.48, 0.16, 0.13, 0.32])
    ax_roc = fig.add_axes([0.1, 0.25, 0.12, 0.6])
    ax_learning_curve = fig.add_axes([0.72, 0.22, 0.25, 0.63])

    # ROC curve combined all data
    team_name = [
        "abethe",
        "amicie",
        "ayoub.ghriss",
        "mk",
        "nguigui",
        "pearrr",
        "Slasnista",
        "vzantedeschi",
        "wwwwmmmm",
    ]
    modality_type = [
        "anatomy",
        "functional",
        "anatomy_functional",
        "functional_age_sex",
        "anatomy_functional_age_sex",
    ]
    all_submissions = [
        tn + "_" + mt for tn, mt in product(team_name, modality_type)
    ]

    y_true_train, y_pred_train, y_true_test, y_pred_test = zip(
        *[load_train_test_prediction(sub) for sub in all_submissions]
    )
    df = pd.DataFrame(
        {
            "y_true_train": y_true_train,
            "y_pred_train": y_pred_train,
            "y_true_test": y_true_test,
            "y_pred_test": y_pred_test,
        },
        index=all_submissions,
    )
    df.index = df.index.str.split("_", n=1, expand=True)
    df = df.reset_index()
    df = df.rename(columns={"level_0": "team", "level_1": "modality"})
    df["modality"] = df["modality"].str.replace("_", " + ")
    df = df[df["modality"] == "anatomy + functional + age + sex"]

    # Seaborn creates too much padding here
    plt.rcParams["ytick.major.pad"] = 2.5
    plt.rcParams["xtick.major.pad"] = 2.5
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 16

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for df_row in df.iterrows():
        fpr, tpr, thres = roc_curve(
            df_row[1]["y_true_test"], df_row[1]["y_pred_test"][:, 1]
        )
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plot the roc for each submission
        ax_roc.plot(fpr, tpr, ":", alpha=0.5, lw=1.5, color="tab:blue")
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plot the mean AUC
    ax_roc.plot(
        mean_fpr, mean_tpr, color="tab:blue", label="Average\naccuracy",
        alpha=0.8
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # plot the chance line
    ax_roc.plot(
        [0, 1], [0, 1], color=".2", linestyle="--", lw=2, alpha=0.8,
        label="Chance"
    )

    # plot the screening control point
    screening_tpr = 0.85
    screening_idx = np.argmax(mean_tpr >= screening_tpr)
    ax_roc.plot(
        [0, mean_fpr[screening_idx]],
        [screening_tpr, screening_tpr],
        color="tab:green",
        linestyle="--",
        alpha=0.8,
        lw=2,
    )
    ax_roc.plot(
        [mean_fpr[screening_idx], mean_fpr[screening_idx]],
        [screening_tpr, 0.05],
        color="tab:green",
        linestyle="--",
        alpha=0.8,
        lw=2,
    )
    ax_roc.text(
        -0.1,
        0.75,
        "Screening\n(85%)",
        transform=ax_roc.transAxes,
        size=16,
        color="tab:green",
        horizontalalignment="right",
    )
    confirmatory_tpr = 0.2
    confirmatory_idx = np.argmax(mean_tpr >= confirmatory_tpr)
    ax_roc.plot(
        [0, mean_fpr[confirmatory_idx]],
        [confirmatory_tpr, confirmatory_tpr],
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        lw=2,
    )
    ax_roc.plot(
        [mean_fpr[confirmatory_idx], mean_fpr[confirmatory_idx]],
        [confirmatory_tpr, 0.05],
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        lw=2,
    )
    ax_roc.text(
        -0.1,
        0.1,
        "Confirmatory\n(20%)",
        transform=ax_roc.transAxes,
        size=16,
        color="tab:red",
        horizontalalignment="right",
    )

    ax_roc.axis("square")
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    ax_roc.set_xlabel("False Positive Rate", size=20)
    ax_roc.set_ylabel(
        "True\nPositive\nRate",
        size=20,
        rotation="horizontal",
        horizontalalignment="right",
    )
    ax_roc.yaxis.set_label_coords(-0.355, 0.335)
    ax_roc.set_xticks([0.0, 0.5, 1.0])
    ax_roc.set_yticks([0.0, 0.5, 1.0])
    ax_roc.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x))
    )
    ax_roc.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x))
    )
    ax_roc.set_title(
        "Prediction accuracy\n", size=20, fontweight="bold", y=1.12
    )
    ax_roc.text(
        0.11,
        1.18,
        "ROC curve analysis",
        size=16,
        va="center",
        transform=ax_roc.transAxes,
    )
    ax_roc.legend(
        loc=(0.53, 0.04),
        frameon=False,
        handlelength=1,
        handletextpad=0.5,
        prop={"size": 14},
    )
    ax_roc.text(
        0.53,
        0.40,
        "AUC={:0.2f}$\\pm${:0.2f}".format(mean_auc, std_auc),
        transform=ax_roc.transAxes,
        size=14,
    )
    sns.despine(ax=ax_roc, offset=10)
    ax_roc.text(
        0.43,
        -0.45,
        "(a)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_roc.transAxes,
    )
    ax_roc.tick_params(length=6)
    ax_roc.text(
        -0.02,
        -0.03,
        "{:2.0f}%".format(mean_fpr[confirmatory_idx] * 100),
        transform=ax_roc.transAxes,
        size=16,
        color="tab:red",
    )
    ax_roc.text(
        0.43,
        -0.03,
        "{:2.0f}%".format(mean_fpr[screening_idx] * 100),
        transform=ax_roc.transAxes,
        size=16,
        color="tab:green",
    )

    # Overall results modalities
    team_name = [
        "abethe",
        "amicie",
        "ayoub.ghriss",
        "mk",
        "nguigui",
        "pearrr",
        "Slasnista",
        "vzantedeschi",
        "wwwwmmmm",
    ]
    modality_type = [
        "anatomy",
        "functional",
        "anatomy_functional",
        "anatomy_functional_age_sex",
    ]
    all_submissions = [
        tn + "_" + mt for tn, mt in product(team_name, modality_type)
    ]

    # compute the ROC-AUC for training and testing and create a dataframe
    roc_auc_train, roc_auc_test = zip(
        *[
            compute_roc_auc_score(*load_train_test_prediction(sub))
            for sub in all_submissions
        ]
    )
    df_roc_auc = pd.DataFrame(
        {"ROC AUC train": roc_auc_train, "ROC AUC test": roc_auc_test},
        index=all_submissions,
    )
    # create a separate column for the team and the modality
    df_roc_auc.index = df_roc_auc.index.str.split("_", n=1, expand=True)
    df_roc_auc = df_roc_auc.reset_index()
    df_roc_auc = df_roc_auc.rename(
        columns={"level_0": "team", "level_1": "modality"}
    )

    # make a plot only with the testing ROC AUC groupy by modality
    # clean the name given to the modality
    df_roc_auc["modality"] = df_roc_auc["modality"].str.replace("_", " + ")
    df_roc_auc["modality"] = df_roc_auc["modality"].replace(
        "anatomy", "anatomy (cortical thickness)"
    )
    df_roc_auc["modality"] = df_roc_auc["modality"].replace(
        "functional", "functional (resting-state fMRI)"
    )

    # Seaborn creates too much padding here
    plt.rcParams["xtick.major.pad"] = 2.5
    plt.rcParams["ytick.labelsize"] = 20
    sns.boxplot(
        data=df_roc_auc, y="modality", x="ROC AUC test", whis=10.0,
        ax=ax_box_plot_1
    )
    for i in range(2):
        ax_box_plot_1.axhspan(2 * i + 0.5, 2 * i + 1.5, color=".9", zorder=0)
    ax_box_plot_1.set_title(
        "Importance of different data modalities", x=-0.5, size=20,
        fontweight="bold"
    )
    ax_box_plot_1.set_xlabel("", size=16)
    ax_box_plot_1.set_ylabel("")
    ax_box_plot_1.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_box_plot_1.set_xticks([0.6, 0.7, 0.8])
    ax_box_plot_1.set_xlim([0.6, 0.8])
    ax_box_plot_1.tick_params(axis="y", which="both", left=False, labelsize=18)
    sns.despine(ax=ax_box_plot_1, offset=10, left=True)
    ax_box_plot_1.text(
        1,
        1,
        "(c)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_box_plot_1.transAxes,
    )
    ax_box_plot_1.tick_params(length=6)
    ax_box_plot_1.set_xlabel("ROC-AUC", size=14)

    # Comparison RDB ABIDE
    team_name = [
        "abethe",
        "amicie",
        "ayoub.ghriss",
        "mk",
        "nguigui",
        "pearrr",
        "Slasnista",
        "vzantedeschi",
        "wwwwmmmm",
    ]
    modality_type = ["anatomy_functional"]
    all_submissions = [
        tn + "_" + mt for tn, mt in product(team_name, modality_type)
    ]

    y_true_train, y_pred_train, y_true_test, y_pred_test = zip(
        *[load_train_test_prediction(sub) for sub in all_submissions]
    )
    df = pd.DataFrame(
        {
            "y_true_train": y_true_train,
            "y_pred_train": y_pred_train,
            "y_true_test": y_true_test,
            "y_pred_test": y_pred_test,
        },
        index=all_submissions,
    )
    df.index = df.index.str.split("_", n=1, expand=True)
    df = df.reset_index()
    df = df.rename(columns={"level_0": "team", "level_1": "modality"})
    df["modality"] = df["modality"].str.replace("_", " + ")

    # Compute the performance only for RDB vs the others
    # Find the index corresponding to RDB
    rdb_idx = np.load("rdb_idx.npy")
    X_test, y_test = get_test_data("..")
    X_test_idx = X_test.index.values
    X_rdb_idx = [X_test_idx == ii for ii in rdb_idx]
    X_rdb_idx = np.vstack(X_rdb_idx)
    X_rdb_idx = np.sum(X_rdb_idx, axis=0).astype(bool)

    auc_rdb = []
    auc_other = []
    for idx, serie in df.iterrows():
        test_data = serie[["y_true_test", "y_pred_test"]]

        y_true_rdb = test_data["y_true_test"][X_rdb_idx]
        y_pred_rdb = test_data["y_pred_test"][X_rdb_idx]
        auc_rdb.append(roc_auc_score(y_true_rdb, y_pred_rdb[:, 1]))

        y_true_other = test_data["y_true_test"][~X_rdb_idx]
        y_pred_other = test_data["y_pred_test"][~X_rdb_idx]
        auc_other.append(roc_auc_score(y_true_other, y_pred_other[:, 1]))

    # Add new column to the dataframe with the AUC
    df["New site"] = auc_rdb
    df["Sites in public data"] = auc_other
    df = df[["team", "modality", "New site", "Sites in public data"]]

    # Seaborn creates too much padding here
    sns.boxplot(
        data=pd.melt(
            df,
            id_vars=["team", "modality"],
            value_vars=["Sites in public data", "New site"],
        ),
        y="variable",
        x="value",
        # hue='modality',
        whis=10.0,
        ax=ax_box_plot_2,
    )
    for i in range(2):
        ax_box_plot_2.axhspan(2 * i + 0.5, 2 * i + 1.5, color=".9", zorder=0)
    ax_box_plot_2.set_title(
        "Heterogeneity across sites", x=-0.88, size=20, fontweight="bold"
    )
    ax_box_plot_2.set_ylabel("")
    ax_box_plot_2.set_xlabel("")
    ax_box_plot_2.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_box_plot_2.set_xticks([0.72, 0.76, 0.8])
    ax_box_plot_2.set_xlim([0.72, 0.8])
    ax_box_plot_2.tick_params(axis="y", which="both", left=False, labelsize=18)
    sns.despine(ax=ax_box_plot_2, offset=10, left=True)
    ax_box_plot_2.text(
        1,
        1,
        "(b)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_box_plot_2.transAxes,
    )
    ax_box_plot_2.tick_params(length=6)

    # Learning curve
    participants = [
        "abethe",
        "amicie",
        "ayoub.ghriss",
        "mk",
        "nguigui",
        "pearrr",
        "Slasnista",
        "vzantedeschi",
        "wwwwmmmm",
    ]
    auc_participants = {}
    for p in participants:
        path_permutation = os.path.join(
            "..",
            "submissions",
            "{}_learning_curve".format(p),
            "training_output",
            "learning_curve_permutation.joblib",
        )

        predictions = joblib.load(path_permutation)

        samples_list = []
        bootstrap_list = []
        auc_list = []
        for n_samples, bootstrap_idx, (y_test, y_pred) in predictions:
            auc = roc_auc_score(y_test, y_pred[:, 1])
            samples_list.append(n_samples)
            bootstrap_list.append(bootstrap_idx)
            auc_list.append(auc)
            auc_participants[p] = auc_list

    auc_participants["n_samples"] = samples_list
    # auc_participants['bootstrap_idx'] = bootstrap_list
    df = pd.DataFrame(auc_participants)
    df = df.groupby("n_samples").mean().unstack().to_frame().reset_index()
    df = df.rename(
        columns={"level_0": "submission", "n_samples": "# subjects",
                 0: "ROC-AUC test"}
    )
    df = df.set_index("# subjects")
    # The sample at 1118 is not useful.
    df = df[~(df.index == 1118)]

    # define the function which will be used to fit the learning curve
    def fit_func(x, a, b):
        return 0.5 + a * (1 - np.exp(-b * np.sqrt(x)))

    # Seaborn creates too much padding here
    plt.rcParams["ytick.major.pad"] = 2.5
    plt.rcParams["xtick.major.pad"] = 2.5
    # Fit an extrapolation function
    max_lr_idx = df.index.max()
    popt, pcov = curve_fit(
        fit_func, df.index.values, df["ROC-AUC test"].values,
        p0=[0.3, 1.0 / max_lr_idx]
    )
    x_test = np.linspace(500, 3000, 200)
    ax_learning_curve.plot(
        x_test, fit_func(x_test, *popt), "r--", label=r"fit:"
    )
    # Bootstrap that fit:
    rng = np.random.RandomState(42)
    bst_popt = list()
    bst_y_test = list()
    for i in range(1000):
        # Bootstrap 1000 times
        choice = rng.randint(low=0, high=df.shape[0], size=df.shape[0])
        this_df = df.iloc[choice]
        this_popt, this_pcov = curve_fit(
            fit_func,
            this_df.index.values,
            this_df["ROC-AUC test"].values,
            p0=[0.3, 1.0 / max_lr_idx],
        )
        bst_popt.append(this_popt)
        bst_y_test.append(fit_func(x_test, *this_popt))
    bst_y_test = np.array(bst_y_test)
    ax_learning_curve.fill_between(
        x_test,
        stats.scoreatpercentile(bst_y_test, 5, axis=0),
        stats.scoreatpercentile(bst_y_test, 95, axis=0),
        alpha=0.15,
        color="r",
    )
    # Plot the data points
    mean_lr = df.groupby("# subjects")["ROC-AUC test"].mean()
    std_lr = df.groupby("# subjects")["ROC-AUC test"].std()
    ax_learning_curve.fill_between(
        std_lr.index,
        mean_lr - std_lr,
        mean_lr + std_lr,
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax_learning_curve.plot(mean_lr.index.values, mean_lr.values, "o-")
    # Decorate
    ax_learning_curve.axhline(
        0.5 + popt[0], color="r", linewidth=1, linestyle=":"
    )
    ax_learning_curve.text(
        2500,
        0.5 + popt[0] - 0.001,
        "AUC = {:.2f}".format(0.5 + popt[0]),
        color="r",
        size=13,
        va="top",
        ha="right",
    )
    sns.despine(offset=10, ax=ax_learning_curve)
    ax_learning_curve.set_ylabel("Prediction performance\n", size=20)
    ax_learning_curve.text(
        -0.23,
        0.5,
        "(ROC-AUC)",
        size=14,
        va="center",
        transform=ax_learning_curve.transAxes,
        rotation=90,
    )
    ax_learning_curve.set_xlabel("Number of subjects in training set", size=20)
    ax_learning_curve.set_xlim([500, 2500])
    ax_learning_curve.set_xticks(np.arange(500, 2500, 500))
    ax_learning_curve.legend(frameon=False, loc=(0.2, -0.02))
    ax_learning_curve.text(
        0.5,
        0.185,
        "$0.5 + %.2f \cdot (1 - e^{- %.3f \sqrt{n}})$" % tuple(popt),
        size=13,
        transform=ax_learning_curve.transAxes,
    )
    ax_learning_curve.set_title(
        "Prediction for various samples sizes\n", size=20, fontweight="bold",
        x=0.5
    )
    ax_learning_curve.set_yticks([0.7, 0.75, 0.8])
    ax_learning_curve.set_ylim([0.7, 0.8])
    ax_learning_curve.text(
        -0.15,
        -0.2,
        "(d)",
        fontweight="bold",
        size=16,
        va="center",
        transform=ax_learning_curve.transAxes,
    )
    ax_learning_curve.text(
        0.35,
        1.04,
        "learning curve",
        size=16,
        va="center",
        transform=ax_learning_curve.transAxes,
    )
    ax_learning_curve.tick_params(length=6)
    ax_learning_curve.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: "{:.2f}".format(x))
    )
    plt.savefig("../figures/single_figure.svg")
