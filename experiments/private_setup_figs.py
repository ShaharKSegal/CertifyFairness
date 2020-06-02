import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import dataset.datasets as ds


def get_sample_size(efg, epsilon, delta):
    if efg >= epsilon:
        return None
    diff = abs(efg - epsilon)
    return math.ceil((2 / diff ** 2) * math.log(2 / delta))


def create_private_test_figure():
    plt.rcParams['font.size'] = 11

    eps_lst = [0.05, 0.075, 0.1]
    efgs = np.linspace(0, max(eps_lst), 100)

    experiments = {"C-MNIST": (0.001, 5000),
                   "C-MNIST bias": (0.450, 5000),
                   "CelebA": (0.0075, 14902),
                   "CelebA bias": (0.034, 14902),
                   "UTKFace": (0.012, 2310),
                   "UTKFace bias": (0.0925, 2310),
                   "Adult Income": (0.108, 5421),
                   "LFW": (0.050, 2519),
                   "TIMIT": (0.040, 1058)}
    experiments_arr = np.array(list(experiments.values()))

    fig, ax = plt.subplots()

    for eps in eps_lst:
        m = np.zeros(efgs.shape)
        for i, efg in enumerate(efgs):
            m[i] = get_sample_size(efg, eps, 0.05)
        ax.plot(efgs, m, color="black", linestyle='dashed', label=r"$\epsilon$={eps}")

    for i, key in enumerate(experiments.keys()):
        ax.annotate(key, (experiments_arr[i, 0] + 0.004, experiments_arr[i, 1] - 300),
                    backgroundcolor="w", fontweight="bold")
    ax.scatter(experiments_arr[:, 0], experiments_arr[:, 1])

    ax.text(0.02, 10000, r'$\epsilon$ = 0.05', ha='center', va='center', rotation=75)
    ax.text(0.045, 10000, r'$\epsilon$ = 0.075', ha='center', va='center', rotation=75)
    ax.text(0.07, 10000, r'$\epsilon$ = 0.1', ha='center', va='center', rotation=75)

    plt.xlabel("Empirical Fairness Gap")
    plt.ylabel(r"Minimal $m_g$")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    plt.xlim([0, 0.15])
    plt.ylim([0, 20000])
    plt.tight_layout()
    plt.show()
    # plt.savefig("private_summary_plot.png")


def get_ODIN_data():
    with open("saved_runs/thresholds_dict.pickle", "rb") as f:
        d = pickle.load(f)

    in_percent = 0.95

    best_key = list(d.keys())[0]
    best_ood = best_th = 0
    res_d = {}
    for key in d:
        a = np.array(d[key])
        in_dist = a[a[:, 1] == 0, 0]
        out_dist = a[a[:, 1] == 1, 0]
        thresholds = np.sort(a[:, 0])

        points = np.zeros((thresholds.shape[0], 2))
        for i, th in enumerate(thresholds):
            in_mask = in_dist <= th
            out_mask = out_dist <= th
            points[i, 0] = in_mask.sum() / in_dist.shape[0]
            points[i, 1] = out_mask.sum() / out_dist.shape[0]  # (out_mask.sum() / out_dist.shape[0])
        res_d[key] = (thresholds, points)
        # find best
        idx = np.argmax(points[:, 0] > in_percent)
        if best_ood < points[idx, 1]:
            best_key, best_ood, best_th = key, points[idx, 1], thresholds[idx]
    thresholds, points = res_d[best_key]
    return thresholds, points, best_key, best_ood, best_th

def get_knn_data():
    df = pd.read_pickle("../preprocess_utils/augment_testing_1_nn_df.pkl")
    k = 1

    groups, groups_count = np.unique(df['fairness_group'].values, return_counts=True)
    groups = groups.astype('int')
    distances = np.sort(df['1_nn_distance'].unique())
    accuracy = np.zeros(distances.shape)
    group_accuracy = np.zeros((*distances.shape, *groups.shape))
    for i, distance in enumerate(distances):
        mask = df['1_nn_distance'] <= distance
        accuracy[i] += (df[mask]['label'] == df[mask]['1_nn_prediction']).sum()
        accuracy[i] += (df[~mask]['label'] == df[~mask]['model_prediction']).sum()
        for j, group in enumerate(groups):
            group_mask = df['fairness_group'] == group
            group_accuracy[i, j] += (df[mask & group_mask]['label'] == df[mask & group_mask]['1_nn_prediction']).sum()
            group_accuracy[i, j] += (
                    df[~mask & group_mask]['label'] == df[~mask & group_mask]['model_prediction']).sum()
    # plot_path = 'augment_testing_{k}-nn_plot_accuracy.png'
    accuracy *= 100 / df.shape[0]
    group_accuracy *= 100 / groups_count
    return distances, groups, accuracy, group_accuracy

def create_knn_ODIN_figure():
    thresholds, points, best_key, best_ood, best_th = get_ODIN_data()
    distances, groups, accuracy, group_accuracy = get_knn_data()

    plt.clf()
    plt.rcParams['font.size'] = 11
    plt.rcParams["figure.figsize"] = (10, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # 1-nn acc
    ax1.set(xlabel=f"{k}-nn distance threshold", ylabel="accuracy")
    ax1.set_title("(a) 1-nn attack accuracy")
    # ax1.set_xlim(left=0.0)
    ax1.plot(distances, accuracy, label='total acc.')
    for j, group in enumerate(groups):
        ax1.plot(distances, group_accuracy[:, j], label=f'$g_{group}$ acc.')
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax1.legend()

    # 1-nn efg
    ax2.set(xlabel=f"{k}-nn distance threshold", ylabel="EFG")
    ax2.set_title("(b) 1-nn attack EFG")
    # ax2.set_xlim(left=0.0)
    ax2.plot(distances, np.abs(group_accuracy[:, 0] - group_accuracy[:, 1]) / 100)
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    #ODIN
    ax3.plot(thresholds, points[:, 0], '--', label="train set")
    ax3.plot(thresholds, points[:, 1], '-', label="test set")
    ax3.set(xlabel=r"$\delta$ Threshold", ylabel=r"Out-of-distribution rate")
    ax3.set_title("(c) ODIN out-of-distribution rate")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("augment_knn_odin.png")
    plt.close()

create_knn_ODIN_figure()