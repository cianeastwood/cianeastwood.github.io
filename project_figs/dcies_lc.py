import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
import seaborn as sns
from tueplots import bundles, figsizes


SEED = 1234

plt.rcParams.update(bundles.icml2022())
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times,amsmath, amsfonts}')


def lc_curve():
    # Settings
    fontsizes = {'font.size': 10,
                 'axes.labelsize': 10,
                 'legend.fontsize': 10,
                 'xtick.labelsize': 10,
                 'ytick.labelsize': 11}
    plt.rcParams.update(fontsizes)
    color = "tab:blue"
    loss_baseline_eps = 0.1

    # Create synthetic data
    capacities = list(range(1, 6))
    losses = [1. / math.exp(c) for c in capacities[:-1]]
    losses += [losses[-1] + 0.02]

    # Plot
    fs = figsizes.aistats2022_half()["figure.figsize"]
    fig, ax = plt.subplots(figsize=(2.5,2.5))
    # Lowest-loss line
    sns.lineplot(x=capacities, y=[losses[-2]] * len(losses), ax=ax, color="black")
    ax.lines[0].set_linestyle("--")
    # Loss-capacity curve
    sns.lineplot(x=capacities, y=losses, ax=ax, color="black", marker="o")

    # Colour area under loss-capacity curve
    xs = np.linspace(capacities[0], capacities[-2], 20)
    ys = np.interp(xs, capacities[:-1], losses[:-1])
    plt.fill_between(xs, ys, [losses[-2]] * len(ys), color=color, alpha=0.75)

    # Colour area under triangle / denominator
    xs = np.linspace(capacities[0], capacities[-1], 20)
    ys = np.interp(xs, [capacities[0], capacities[-1]], [losses[0] + loss_baseline_eps, losses[-2]])
    plt.fill_between(xs, ys, [losses[-2]] * len(ys), color=color, alpha=0.15)

    # Add legend
    custom_lines = [Line2D([0], [0], color=color, lw=5, alpha=0.75),
                    Line2D([0], [0], color=color, lw=5, alpha=0.15)]
    ax.legend(custom_lines, ['AULCC', 'Normalizer'], handlelength=1)

    # Create tick labels
    c_labels = [f"$\\kappa_{i}$" for i in capacities]
    c_labels[-2] = "$\\kappa_*$"
    l_ticks = [losses[0] + loss_baseline_eps] + losses[0:1] + losses[-2:-1]
    l_labels = [r"$\ell^b$", r"$\ell^1$", r"$\ell^*$"]

    # Final settings
    ax.set_xlim(min(capacities) - 0.4, max(capacities) + 0.2)
    ax.set_ylim(min(losses) - 0.1, max(losses) + 0.075 + loss_baseline_eps)
    ax.set_xticks(capacities)
    ax.set_xticklabels(c_labels)
    ax.set_yticks(l_ticks)
    ax.set_yticklabels(l_labels)
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Loss")

    # Save
    plt.savefig("lc_curve.pdf")


def lc_curve_detailed():
    # Settings
    fontsizes = {'font.size': 10,
                 'axes.labelsize': 10,
                 'legend.fontsize': 10,
                 'xtick.labelsize': 10,
                 'ytick.labelsize': 11}
    plt.rcParams.update(fontsizes)
    color = "tab:blue"
    loss_baseline_eps = 0.07

    # Create synthetic data
    capacities = list(range(1, 6))
    losses = [1. / math.exp(c) for c in capacities[:-1]]
    losses += [losses[-1] + 0.02]
    lowest_probe_loss = losses[-2] - 0.05

    # Plot
    fs = figsizes.aistats2022_half()["figure.figsize"]
    fs = (fs[0] * 0.75, fs[1])
    fig, ax = plt.subplots(figsize=fs)
    # Lowest-loss line
    sns.lineplot(x=capacities, y=[losses[-2]] * len(losses), ax=ax, color="black")
    ax.lines[0].set_linestyle("--")
    # Loss-capacity curve
    sns.lineplot(x=capacities, y=losses, ax=ax, color="black", marker="o")

    # Colour area under loss-capacity curve
    xs = np.linspace(capacities[0], capacities[-2], 20)
    ys = np.interp(xs, capacities[:-1], losses[:-1])
    plt.fill_between(xs, ys, [losses[-2]] * len(ys), color=color, alpha=0.75)

    # Colour area under triangle / denominator
    xs = np.linspace(capacities[0], capacities[-1], 20)
    ys = np.interp(xs, [capacities[0], capacities[-1]], [losses[0] + loss_baseline_eps, lowest_probe_loss])
    plt.fill_between(xs, ys, [lowest_probe_loss] * len(ys), color=color, alpha=0.15)

    # Add legend
    custom_lines = [Line2D([0], [0], color=color, lw=5, alpha=0.75),
                    Line2D([0], [0], color=color, lw=5, alpha=0.15)]
    ax.legend(custom_lines, ['AULCC', 'Normalizer'], handlelength=1)

    # Create tick labels
    c_labels = [f"$\\kappa_{i}$" for i in capacities]
    c_labels[-2] = "$\\kappa_{*, \mathbf{c}}$"
    l_ticks = [losses[0] + loss_baseline_eps] + losses[0:1] + losses[1:2] + losses[-2:-1] + [lowest_probe_loss]
    l_labels = [r"$\ell^b$", r"$\ell^{1, \mathbf{c}}$", r"$\ell^{2, \mathbf{c}}$", r"$\ell^{*,\mathbf{c}}$", r"$\ell^*$"]

    # Final settings
    ax.set_xlim(min(capacities) - 0.4, max(capacities) + 0.2)
    ax.set_ylim(min(losses) - 0.1, max(losses) + 0.075 + loss_baseline_eps)
    ax.set_xticks(capacities)
    ax.set_xticklabels(c_labels)
    ax.set_yticks(l_ticks)
    ax.set_yticklabels(l_labels)
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Loss", labelpad=0)

    # Save
    plt.savefig("lc_curve_narrow.pdf")


if __name__ == "__main__":
    np.random.seed(SEED)
    lc_curve()
